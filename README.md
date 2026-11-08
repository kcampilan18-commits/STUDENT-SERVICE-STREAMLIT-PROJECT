# Student Service Request Monitoring System — Section B

## Project purpose
A Streamlit decision-support application for monitoring student service requests and identifying possible Service-Level Agreement (SLA) breaches. It reuses the Section B midterm target and submission-time predictor design.

## Important model artifact requirement
The provided midterm code trains `decision_tree` in memory but does **not** save the fitted pipeline. The source PDF requires a saved fitted preprocessing + model pipeline at `models/trained_model.joblib`. The actual CSV dataset was not included with the current files, so no genuine trained artifact or dataset-specific metric values can be generated here without inventing results.

Run the following cell at the end of your original Colab notebook, **after** `decision_tree.fit(X_train, y_train)`:

```python
import os, joblib
os.makedirs("models", exist_ok=True)
joblib.dump(decision_tree, "trained_model.joblib")
from google.colab import files
files.download("trained_model.joblib")
```

Place the downloaded file in this project's `models/` folder and commit it to the private repository only if permitted by your instructor. Do not commit student records or credentials.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.
5. Upload the instructor-provided CSV in Dataset, Data Analysis, or Dashboard.

## Pages
- Home — project purpose and navigation.
- Dataset — CSV upload, required-column validation, preview, missing/duplicate summary.
- Data Analysis — descriptive statistics, class distribution, and interactive charts.
- Prediction — loads the saved joblib pipeline and accepts submission-time feature values.
- Dashboard — workload KPIs, historical SLA patterns, pending review queue, human follow-up.
- System Design — actors, functional/non-functional requirements, use case diagram, analytical workflow.

## Midterm model definition preserved
- Target: `SLA_Breached`
- Label 1: `Processing_Days > SLA_Days` (completed requests with both values present)
- Label 0: no breach
- Candidate features: `Service_Type`, `Requester_Type`, `Submission_Date`, `Submission_Weekday`, `Requested_Channel`, `Priority`, `Assigned_Office`, `Queue_Length`, `Queue_Category`, `Required_Documents_Complete`, `SLA_Days`
- Excluded leakage/identifier fields: `Request_ID`, `Processing_Days`, `Status`, `Reopened`, `Satisfaction_Score`, `SLA_Breached`
- Estimator: `DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=5, class_weight="balanced")`
- Preprocessing: median imputation for numeric fields; most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")` for categorical fields.
- Train/test split: 80/20, random state 42, stratified where possible.

**Date feature note:** the original notebook converts `Submission_Date` to ordinal before fitting. The prediction page does the same. Keep the training and inference feature types consistent.

## Testing checklist
- [ ] App opens with no errors.
- [ ] Valid CSV uploads and required columns are validated.
- [ ] Missing columns and malformed CSV produce understandable errors.
- [ ] Empty dataset and missing values are handled.
- [ ] At least three charts render with the uploaded dataset.
- [ ] Saved model loads without retraining on page refresh.
- [ ] Prediction uses the same feature names and transformations as the fitted pipeline.
- [ ] Five sample predictions are tested and screenshots captured.
- [ ] Metrics are copied from the actual Colab evaluation output (do not invent).
- [ ] Human review warning is visible.
- [ ] Use case and workflow diagrams match the app.
- [ ] Deployment URL and GitHub URL are added to the final report.

## Five sample predictions
Use five synthetic examples or properly authorized records. Record the input values, predicted class, and breach probability only if `predict_proba` supports it. Do not fabricate prediction results. The app returns one result per submitted input; capture five separate test runs or extend it to accept a batch file.

## Deployment
Push the project to GitHub. On Streamlit Community Cloud, create an app from the repository, select the branch, and set `app.py` as the entry point. Add the deployed URL to your report. Use synthetic or properly authorized data for public deployment.

## Limitations
- Model reliability depends on the original dataset, class balance, missingness, and representative historical records.
- The prediction target is labeled only for completed requests, so the model is trained on historical completed records.
- The dashboard's historical SLA breach rate is not a prediction for currently pending requests.
- Predictions are advisory; authorized staff make final decisions.
