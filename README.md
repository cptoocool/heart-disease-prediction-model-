# 🫀 CardioCare: Heart Disease Prediction App (Streamlit & Scikit-Learn)

An end-to-end Machine Learning web application designed to predict the likelihood of heart disease in patients using clinical biomarkers.

---

## 🌟 Key Features

1. **🩺 Patient Risk Assessment:**
   - Interactive inputs with medical units, clinical ranges, and preset quick-load profiles (*Healthy*, *Borderline*, *High Risk*).
   - Real-time prediction gauge with risk probability (%) and tiered classification (*Low Risk*, *Moderate Risk*, *High Risk*).
   - Dynamic clinical risk factor breakdown (identifying ST depression, ST slope abnormalities, angina, hypertension, etc.).
   - Personalized medical insights and lifestyle suggestions.

2. **📊 Model Benchmarking & Multi-Algorithm Support:**
   - 5 models trained and evaluated with 5-fold cross-validation:
     - **Gradient Boosting Classifier** (Top F1: ~0.898, Accuracy: ~88.6%, ROC-AUC: 0.916)
     - **Logistic Regression** (ROC-AUC: ~0.934, Accuracy: ~88.0%)
     - **Support Vector Machine (SVC)** (ROC-AUC: ~0.937, Accuracy: ~86.4%)
     - **Random Forest Classifier** (CV Acc: ~86.4%, ROC-AUC: ~0.925)
     - **K-Nearest Neighbors (KNN)** (Accuracy: ~86.4%, ROC-AUC: ~0.931)
   - Interactive model switcher directly in the sidebar.
   - Interactive confusion matrices and Random Forest feature importance rankings.

3. **📈 Dataset Analytics & Visualizations (EDA):**
   - Exploratory distributions of Age, Cholesterol, Max Heart Rate, Chest Pain types, and ST Slope.
   - Raw dataset table inspector with summary statistics.

4. **📁 Batch Patient Classification:**
   - Upload multiple patient records in CSV format.
   - One-click template download.
   - Bulk prediction generation with downloadable annotated CSV reports.

---

## 🚀 How to Run the App

### Option 1: Using your Anaconda Python environment (Recommended)
From the project folder:
```bash
/opt/anaconda3/bin/streamlit run app.py
```

### Option 2: Using any Python environment with `streamlit`
```bash
cd /Users/chhaterpaldangi/.gemini/antigravity/scratch/heart_disease_prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Dataset Information
- **Total Patients:** 918
- **Features:** 11 clinical biomarkers (Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope)
- **Target:** `HeartDisease` (1 = Presence of heart disease, 0 = Normal)
- **Handling of Implausible Zeros:** 0 values in `Cholesterol` and `RestingBP` imputed using non-zero cohort means.
- **Scaling & Encoding:** `StandardScaler` for continuous features, `OneHotEncoder(drop='first')` for categorical features, bundled in Scikit-Learn `ColumnTransformer` pipelines.

---

*Disclaimer: For demonstration, education, and research purposes only. Not intended for direct medical diagnosis.*
