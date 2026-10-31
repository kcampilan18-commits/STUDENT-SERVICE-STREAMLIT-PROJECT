import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Student Service Request Monitoring", layout="wide")
MODEL_PATH = Path(__file__).parent / "models" / "trained_model.joblib"
DATA_PATH = Path(__file__).parent / "student_service_requests_cleaned.csv"

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_default_data():
    return pd.read_csv(DATA_PATH)

bundle = load_bundle()
model = bundle["pipeline"]
features = bundle["feature_columns"]
metrics = bundle["metrics"]

st.sidebar.title("Student Service Monitoring")
page = st.sidebar.selectbox("Module", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])

def prep(df):
    df = df.copy()
    if "SLA_Breach" in df.columns:
        df["SLA_Breach"] = df["SLA_Breach"].astype(str).str.strip().str.lower().map({"yes":1,"no":0})
    return df

if page == "Home":
    st.title("Student Service Request Monitoring System")
    st.write("Decision-support tool for monitoring request workload and identifying possible SLA-breach risk.")
    st.info("Model flags are advisory. Authorized staff must review request details before making service decisions.")
    st.subheader("Model evaluation (held-out test set)")
    st.json(metrics)
    st.write("Model: Decision Tree pipeline; trained once and loaded from models/trained_model.joblib.")
elif page == "Dataset":
    st.title("Dataset Upload and Validation")
    uploaded = st.file_uploader("Upload a student service request CSV", type=["csv"])
    try:
        df = pd.read_csv(uploaded) if uploaded else load_default_data()
        st.caption("Showing uploaded data." if uploaded else "Showing provided cleaned dataset.")
        st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error("Required model features missing: " + ", ".join(missing))
        else:
            st.success("Required model features are present.")
        c1,c2 = st.columns(2)
        c1.metric("Missing cells", int(df.isna().sum().sum()))
        c2.metric("Duplicate rows", int(df.duplicated().sum()))
        st.dataframe(df.head(50), use_container_width=True)
        st.subheader("Missing values by column")
        st.dataframe(df.isna().sum().rename("Missing count"))
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
elif page == "Data Analysis":
    st.title("Exploratory Data Analysis")
    df = load_default_data()
    a,b,c = st.columns(3)
    a.metric("Requests", len(df))
    if "SLA_Breach" in df:
        rate = df["SLA_Breach"].astype(str).str.lower().eq("yes").mean()*100
        b.metric("Observed SLA breach rate", f"{rate:.1f}%")
    if "Queue_Length_At_Submission" in df:
        c.metric("Average queue length", f"{df['Queue_Length_At_Submission'].mean():.1f}")
    st.subheader("Request volume by service type")
    st.bar_chart(df["Service_Type"].value_counts())
    st.subheader("Observed SLA breach distribution")
    st.bar_chart(df["SLA_Breach"].value_counts())
    if "Submission_Day" in df:
        st.subheader("Requests by submission day")
        st.bar_chart(df["Submission_Day"].value_counts())
    st.subheader("Descriptive statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)
elif page == "Prediction":
    st.title("Predict SLA-Breach Risk")
    st.write("Enter values available at request submission. The model predicts the class label; it does not authorize denial of service.")
    df = load_default_data()
    values = {}
    with st.form("prediction_form"):
        cols = st.columns(2)
        for i, col in enumerate(features):
            series = df[col]
            with cols[i % 2]:
                if pd.api.types.is_numeric_dtype(series):
                    val = float(series.median()) if series.notna().any() else 0.0
                    values[col] = st.number_input(col, value=val)
                else:
                    opts = sorted(series.dropna().astype(str).unique().tolist())
                    values[col] = st.selectbox(col, opts) if opts else st.text_input(col)
        submitted = st.form_submit_button("Predict")
    if submitted:
        row = pd.DataFrame([values], columns=features)
        result = int(model.predict(row)[0])
        st.subheader("Prediction result")
        st.metric("Predicted SLA breach", "Yes — review" if result == 1 else "No — routine review")
        if hasattr(model, "predict_proba"):
            st.caption(f"Model estimated breach probability: {model.predict_proba(row)[0][1]:.1%}")
        st.warning("This is a model estimate, not a final decision. Verify the request and record an authorized human decision.")
elif page == "Dashboard":
    st.title("Decision-Support Dashboard")
    df = load_default_data()
    breach = df["SLA_Breach"].astype(str).str.lower().eq("yes")
    c1,c2,c3 = st.columns(3)
    c1.metric("Total requests", len(df))
    c2.metric("Observed breaches", int(breach.sum()))
    c3.metric("Observed breach rate", f"{breach.mean()*100:.1f}%")
    st.subheader("Queue workload")
    if "Queue_Length_At_Submission" in df:
        st.bar_chart(df.groupby("Service_Type")["Queue_Length_At_Submission"].mean())
    st.subheader("Review queue — historical breach examples")
    st.dataframe(df.loc[breach].head(30), use_container_width=True)
    st.caption("Historical examples are not live predictions. Staff should verify current request status.")
else:
    st.title("System Design")
    st.subheader("Actors")
    st.write("- Student Services Staff: review requests, queue workload, and risk flags.\n- Administrator: manage access, inspect analytics, and maintain the system.")
    st.subheader("Functional requirements")
    for req in [
        "The system shall accept CSV uploads and validate required model fields.",
        "The system shall display dataset previews, missing values, and duplicate summaries.",
        "The system shall summarize request volume and SLA-breach distribution.",
        "The system shall predict SLA-breach class using the saved Decision Tree pipeline.",
        "The system shall display risk indicators for authorized human review."
    ]: st.write("• " + req)
    st.subheader("Non-functional requirements")
    st.write("- Privacy: use synthetic or properly authorized records.\n- Reliability: validate schema and handle invalid inputs.\n- Usability: show clear labels, limitations, and human-review guidance.")
    st.subheader("Decision workflow")
    st.code("CSV / request inputs → validation → saved model prediction → staff reviews context → staff records final decision")
