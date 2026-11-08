import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def predict_rows(model, frame):
    pred = model.predict(frame)
    out = pd.DataFrame({"prediction": pred})
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(frame)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            out["probability"] = probs[:, classes.index(1)]
    return out

def feature_form(model):
    # Read exact original feature names from the fitted ColumnTransformer.
    pre = model.named_steps["preprocessor"]
    features = []
    for _, _, cols in pre.transformers_:
        features.extend(list(cols))
    features = list(dict.fromkeys(features))
    values = {}
    st.subheader("Enter submission-time information")
    for feature in features:
        if feature == "Submission_Date":
            value = st.date_input(feature)
            values[feature] = pd.Timestamp(value).toordinal()
        elif feature in ["Queue_Length", "SLA_Days"]:
            values[feature] = float(st.number_input(feature, min_value=0.0, value=0.0))
        elif feature == "Submission_Weekday":
            values[feature] = st.selectbox(feature, ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        elif feature == "Queue_Category":
            values[feature] = st.selectbox(feature, ["Low","Medium","High"])
        elif feature == "Required_Documents_Complete":
            values[feature] = st.selectbox(feature, ["Yes","No"])
        else:
            values[feature] = st.text_input(feature, value="")
    return values
