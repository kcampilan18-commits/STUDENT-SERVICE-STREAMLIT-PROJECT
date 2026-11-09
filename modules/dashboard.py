import streamlit as st
import pandas as pd
import plotly.express as px
from modules.data_processing import prepare_analysis_data

def render_dashboard(raw):
    df, eligible = prepare_analysis_data(raw)
    total = len(df)
    pending = int(df["Status"].eq("Pending").sum()) if "Status" in df else 0
    completed = int(eligible.sum())
    breaches = int(df.loc[eligible, "SLA_Breached"].sum()) if completed else 0
    rate = breaches / completed * 100 if completed else 0
    a,b,c,d = st.columns(4)
    a.metric("Total requests", f"{total:,}")
    b.metric("Pending requests", f"{pending:,}")
    c.metric("Eligible completed", f"{completed:,}")
    d.metric("Observed SLA breach rate", f"{rate:.1f}%")
    st.caption("Historical breach rate applies only to completed requests with known processing and SLA days.")
    if "Service_Type" in df:
        counts = df["Service_Type"].value_counts().rename_axis("Service Type").reset_index(name="Requests")
        st.plotly_chart(px.bar(counts, x="Service Type", y="Requests", title="Workload by Service Type"), use_container_width=True)
    if "Assigned_Office" in df and eligible.any():
        office = df.loc[eligible].groupby("Assigned_Office")["SLA_Breached"].mean().mul(100).reset_index(name="Breach Rate (%)")
        st.plotly_chart(px.bar(office, x="Assigned_Office", y="Breach Rate (%)", title="Historical SLA Breach Rate by Office"), use_container_width=True)
    st.subheader("Staff review queue")
    if "Status" in df:
        queue = df.loc[df["Status"].eq("Pending")].copy()
        st.write("Pending requests are listed for operational follow-up. This is not a model-generated risk ranking.")
        safe_cols = [c for c in ["Request_ID","Service_Type","Priority","Assigned_Office","Queue_Length","SLA_Days","Required_Documents_Complete"] if c in queue]
        st.dataframe(queue[safe_cols].head(100), use_container_width=True)
    st.warning("Recommended human follow-up: verify missing documents, confirm queue conditions, check the applicable SLA, and record the staff decision. Do not automatically reject or deny a service request.")
