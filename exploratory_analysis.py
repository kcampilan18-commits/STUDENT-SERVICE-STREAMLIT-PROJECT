import streamlit as st
import pandas as pd
import plotly.express as px
from modules.data_processing import prepare_analysis_data

def render_eda(raw):
    df, eligible = prepare_analysis_data(raw)
    st.subheader("Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
    if "SLA_Breached" in df:
        counts = df.loc[eligible, "SLA_Breached"].value_counts().rename(index={0:"No breach",1:"Breach"})
        st.subheader("SLA breach class distribution")
        if not counts.empty:
            st.plotly_chart(px.pie(values=counts.values, names=counts.index, title="Completed Request SLA Classes"), use_container_width=True)
            st.dataframe(counts.rename("Count").to_frame())
    if "Service_Type" in df:
        svc = df["Service_Type"].value_counts().rename_axis("Service Type").reset_index(name="Requests")
        st.plotly_chart(px.bar(svc, x="Service Type", y="Requests", title="Request Volume by Service Type"), use_container_width=True)
    if "Queue_Category" in df and eligible.any():
        q = df.loc[eligible].groupby("Queue_Category", observed=False)["SLA_Breached"].mean().mul(100).reset_index(name="Breach Rate (%)")
        st.plotly_chart(px.bar(q, x="Queue_Category", y="Breach Rate (%)", title="SLA Breach Rate by Queue Category"), use_container_width=True)
    if "Assigned_Office" in df and eligible.any():
        office = df.loc[eligible].groupby("Assigned_Office")["SLA_Breached"].agg(Requests="count", Breach_Rate="mean").reset_index()
        office["Breach Rate (%)"] = office["Breach_Rate"] * 100
        st.plotly_chart(px.bar(office, x="Assigned_Office", y="Breach Rate (%)", hover_data=["Requests"], title="SLA Breach Rate by Assigned Office"), use_container_width=True)
    st.caption("SLA_Breached is defined only for completed requests with known Processing_Days and SLA_Days: Processing_Days > SLA_Days.")
