import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from modules.data_processing import validate_dataset, summarize_quality, prepare_analysis_data
from modules.exploratory_analysis import render_eda
from modules.prediction import load_model, predict_rows, feature_form
from modules.dashboard import render_dashboard

st.set_page_config(page_title="Student Service Request Monitoring", page_icon="🎓", layout="wide")
st.title("Student Service Request Monitoring System")
st.caption("Section B | SLA breach prediction and student-services decision support")

MODEL_PATH = Path("models/trained_model.joblib")
menu = st.sidebar.selectbox("Module", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])

@st.cache_data(show_spinner=False)
def read_csv(uploaded_file):
    return pd.read_csv(uploaded_file, low_memory=False)

def get_data():
    uploaded = st.session_state.get("dataset_upload")
    if uploaded is not None:
        try:
            return read_csv(uploaded)
        except Exception as exc:
            st.error(f"Unable to read CSV: {exc}")
    return None

if menu == "Home":
    st.header("Home")
    st.write("This application monitors student service requests and uses a saved Decision Tree pipeline to flag possible SLA breaches for staff review.")
    c1, c2 = st.columns(2)
    c1.metric("Prediction target", "SLA_Breached")
    c2.metric("Model", "Decision Tree")
    st.info("Predictions are advisory only. Authorized staff must review the request and make the final service decision.")
    st.subheader("How to use the system")
    st.markdown("1. Upload the student service request CSV in **Dataset**.\\n2. Review data quality and the preview.\\n3. Explore request volume and SLA patterns in **Data Analysis**.\\n4. Use **Prediction** only after placing the fitted model at `models/trained_model.joblib`.\\n5. Review workload and risk indicators in **Dashboard**.")

elif menu == "Dataset":
    st.header("Dataset Upload and Validation")
    uploaded = st.file_uploader("Upload student_service_requests.csv", type=["csv"], key="dataset_upload")
    if uploaded:
        try:
            data = read_csv(uploaded)
            valid, missing, extra = validate_dataset(data)
            if valid:
                st.success("Required columns are present.")
            else:
                st.error("Missing required columns: " + ", ".join(missing))
            if extra:
                st.caption("Additional columns: " + ", ".join(extra))
            st.subheader("Data preview")
            st.dataframe(data.head(20), use_container_width=True)
            st.write(f"Rows: **{len(data):,}** | Columns: **{len(data.columns)}**")
            quality = summarize_quality(data)
            a,b,c = st.columns(3)
            a.metric("Missing cells", int(data.isna().sum().sum()))
            b.metric("Duplicate rows", int(data.duplicated().sum()))
            b2 = data["Request_ID"].duplicated().sum() if "Request_ID" in data else 0
            c.metric("Duplicate Request IDs", int(b2))
            st.subheader("Column quality summary")
            st.dataframe(quality, use_container_width=True)
        except Exception as exc:
            st.error(f"Invalid CSV file: {exc}")
    else:
        st.info("Upload a CSV file to begin. The expected dataset schema is shown below.")
        st.code("Request_ID, Service_Type, Requester_Type, Submission_Date, Requested_Channel, Priority, Assigned_Office, Queue_Length, Required_Documents_Complete, Processing_Days, SLA_Days, Status, Reopened, Satisfaction_Score")

elif menu == "Data Analysis":
    st.header("Exploratory Data Analysis")
    uploaded = st.file_uploader("Upload dataset CSV", type=["csv"], key="eda_upload")
    if uploaded:
        try:
            data = read_csv(uploaded)
            render_eda(data)
        except Exception as exc:
            st.error(f"Could not analyze this file: {exc}")
    else:
        st.info("Upload the dataset to display descriptive statistics, class distribution, and interactive charts.")

elif menu == "Prediction":
    st.header("SLA Breach Prediction")
    if not MODEL_PATH.exists():
        st.warning("The trained pipeline file is not yet available at `models/trained_model.joblib`.")
        st.markdown("Train and export the pipeline once using your midterm dataset. See **README.md** for the exact one-time preparation steps.")
    else:
        model = load_model(str(MODEL_PATH))
        st.success("Saved trained pipeline loaded.")
        st.caption("Model inference uses submission-time features only. Processing_Days, Status, Reopened, Satisfaction_Score, Request_ID, and SLA_Breached are excluded.")
        values = feature_form(model)
        if st.button("Predict SLA breach risk", type="primary"):
            try:
                result = predict_rows(model, pd.DataFrame([values]))
                st.subheader("Prediction result")
                label = int(result["prediction"].iloc[0])
                st.metric("Predicted class", "Possible SLA breach" if label == 1 else "No breach predicted")
                if "probability" in result and pd.notna(result["probability"].iloc[0]):
                    st.metric("Estimated breach probability", f"{result['probability'].iloc[0]:.1%}")
                st.warning("This is a decision-support flag, not an automatic decision. Verify request details and context before acting.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

elif menu == "Dashboard":
    st.header("Decision-Support Dashboard")
    uploaded = st.file_uploader("Upload dataset CSV", type=["csv"], key="dashboard_upload")
    if uploaded:
        try:
            render_dashboard(read_csv(uploaded))
        except Exception as exc:
            st.error(f"Dashboard could not be generated: {exc}")
    else:
        st.info("Upload the dataset to view service volume, queue status, historical SLA performance, and review priorities.")

elif menu == "System Design":
    st.header("Systems Analysis and Design")
    st.subheader("Actors")
    st.markdown("- **Administrator:** views service volume, SLA performance, risk flags, queue status, request details, and exports reports.\\n- **Student Services Staff:** monitors queue status and reviews individual request details and risk flags.")
    st.subheader("Functional requirements")
    requirements = [
        "The system shall accept a CSV dataset and validate required columns before analysis.",
        "The system shall display data previews, missing values, duplicate records, and descriptive statistics.",
        "The system shall display request volume and SLA breach patterns using interactive charts.",
        "The system shall use the saved Decision Tree preprocessing and classification pipeline to predict possible SLA breaches from submission-time features.",
        "The system shall display risk flags and relevant request information to authorized staff for human review."
    ]
    for i, req in enumerate(requirements, 1):
        st.write(f"**FR{i}.** {req}")
    st.subheader("Non-functional requirements")
    st.markdown("- **Security and privacy:** use synthetic or properly authorized records; do not publish sensitive student information.\\n- **Usability:** provide clear labels, validation messages, and understandable risk explanations.\\n- **Reliability:** load a fixed saved model, handle invalid inputs, and show model limitations.")
    st.subheader("Use case diagram")
    st.image("assets/use_case_diagram.png", use_container_width=True)
    st.subheader("Analytical decision workflow")
    st.image("assets/decision_workflow.png", use_container_width=True)
    st.subheader("Human decision rule")
    st.info("A predicted breach is a review prompt. Staff check the request, submitted documents, queue conditions, and relevant context, then make and record the final decision outside the model.")
