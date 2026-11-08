import pandas as pd

EXPECTED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Submission_Date",
    "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
    "Required_Documents_Complete", "Processing_Days", "SLA_Days",
    "Status", "Reopened", "Satisfaction_Score"
]
REQUIRED_COLUMNS = [
    "Request_ID", "Service_Type", "Submission_Date",
    "Queue_Length", "Processing_Days", "SLA_Days", "Status"
]

def validate_dataset(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    return len(missing) == 0, missing, extra

def summarize_quality(df):
    return pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Missing Values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique Values": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })

def prepare_analysis_data(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        df[col] = df[col].replace("", pd.NA)
    if "Request_ID" in df:
        df = df.loc[~(df["Request_ID"].notna() & df["Request_ID"].duplicated())].copy()
    if "Submission_Date" in df:
        df["Submission_Date"] = pd.to_datetime(df["Submission_Date"], errors="coerce")
        df["Submission_Weekday"] = df["Submission_Date"].dt.day_name()
    for col in ["Queue_Length", "Processing_Days", "SLA_Days", "Satisfaction_Score"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Queue_Length" in df:
        df["Queue_Category"] = pd.cut(
            df["Queue_Length"], bins=[-float("inf"), 5, 15, float("inf")],
            labels=["Low", "Medium", "High"]
        )
    if "Status" in df:
        status_map = {"completed":"Completed","complete":"Completed","pending":"Pending",
                      "open":"Pending","cancelled":"Cancelled","canceled":"Cancelled"}
        df["Status"] = df["Status"].map(lambda x: status_map.get(str(x).lower(), x) if pd.notna(x) else x)
    eligible = (
        df.get("Status", pd.Series(index=df.index, dtype="object")).eq("Completed")
        & df.get("Processing_Days", pd.Series(index=df.index, dtype="float")).notna()
        & df.get("SLA_Days", pd.Series(index=df.index, dtype="float")).notna()
    )
    df["SLA_Breached"] = pd.NA
    df.loc[eligible, "SLA_Breached"] = (
        df.loc[eligible, "Processing_Days"] > df.loc[eligible, "SLA_Days"]
    ).astype(int)
    return df, eligible
