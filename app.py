import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CardioCare | Heart Disease Prediction AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #FF4B4B 0%, #B81B24 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.2);
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffebee !important;
        color: #d32f2f !important;
    }
    .risk-badge-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        border: 1px solid #ef9a9a;
    }
    .risk-badge-low {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        border: 1px solid #a5d6a7;
    }
    .risk-badge-medium {
        background-color: #fff8e1;
        color: #f57f17;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        border: 1px solid #ffe082;
    }
    .alert-card {
        padding: 16px;
        border-radius: 8px;
        margin-top: 12px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD SAVED MODEL ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "heart_disease_artifacts.joblib")
    data_path = os.path.join(base_dir, "heart.csv")
    
    if not os.path.exists(model_path):
        st.error(f"Model artifacts not found at {model_path}. Please run train_model.py first.")
        st.stop()
        
    artifacts = joblib.load(model_path)
    df = pd.read_csv(data_path) if os.path.exists(data_path) else None
    return artifacts, df

artifacts, df_dataset = load_artifacts()
pipelines = artifacts['pipelines']
metrics = artifacts['metrics']
best_model_name = artifacts['best_model_name']
top_features = artifacts['feature_importances']

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/anatomical-heart.png", width=80)
    st.title("CardioCare AI")
    st.markdown("Advanced Machine Learning Heart Disease Risk Assessment Tool.")
    
    st.divider()
    st.subheader("⚙️ Model Configuration")
    selected_model_name = st.selectbox(
        "Active Classifier",
        options=list(pipelines.keys()),
        index=list(pipelines.keys()).index(best_model_name),
        help="Select which trained ML model to use for inference."
    )
    
    active_pipeline = pipelines[selected_model_name]
    model_metric = metrics[selected_model_name]
    
    st.markdown(f"**Model Type:** `{selected_model_name}`")
    st.markdown(f"**Test Accuracy:** `{model_metric['accuracy'] * 100:.2f}%`")
    st.markdown(f"**F1 Score:** `{model_metric['f1_score']:.4f}`")
    st.markdown(f"**ROC AUC:** `{model_metric['roc_auc']:.4f}`")
    
    st.divider()
    st.info("💡 **Clinical Tip:** ST_Slope, Oldpeak (ST Depression), and Exercise Angina are the strongest clinical predictors in this cohort.")
    
    st.caption("Developed with Streamlit & Scikit-Learn • Data source: UCI & Kaggle Heart Dataset")

# --- HEADER SECTION ---
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.2rem;">🫀 CardioCare: Heart Disease Prediction System</h1>
    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 1.05rem;">
        Early risk stratification using multi-model machine learning on clinical biomarkers
    </p>
</div>
""", unsafe_allow_html=True)

# --- MAIN NAVIGATION TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Patient Risk Assessment",
    "📊 Model Benchmarks & Comparison",
    "📈 Dataset Analytics & EDA",
    "📁 Batch Prediction"
])

# ==============================================================================
# TAB 1: PATIENT RISK ASSESSMENT
# ==============================================================================
with tab1:
    st.subheader("Individual Patient Health Profile")
    st.markdown("Enter the patient's diagnostic values below to compute personalized heart disease probability.")
    
    # Preset Profiles for Testing
    col_preset, col_clear = st.columns([4, 1])
    with col_preset:
        preset = st.radio(
            "⚡ Quick Preset Example Profiles:",
            ["Default / Custom", "Healthy Profile (Low Risk)", "Borderline Profile (Medium Risk)", "Cardiac Patient Profile (High Risk)"],
            horizontal=True
        )
    
    # Default values based on preset
    if preset == "Healthy Profile (Low Risk)":
        d_age, d_sex, d_cp = 38, "F", "ATA"
        d_bp, d_chol, d_fbs = 118, 175, "Normal (<= 120 mg/dl)"
        d_ecg, d_maxhr, d_exang = "Normal", 172, "No"
        d_oldpeak, d_slope = 0.0, "Up"
    elif preset == "Borderline Profile (Medium Risk)":
        d_age, d_sex, d_cp = 54, "M", "NAP"
        d_bp, d_chol, d_fbs = 135, 230, "Normal (<= 120 mg/dl)"
        d_ecg, d_maxhr, d_exang = "ST", 140, "No"
        d_oldpeak, d_slope = 1.0, "Flat"
    elif preset == "Cardiac Patient Profile (High Risk)":
        d_age, d_sex, d_cp = 62, "M", "ASY"
        d_bp, d_chol, d_fbs = 160, 280, "High (> 120 mg/dl)"
        d_ecg, d_maxhr, d_exang = "LVH", 115, "Yes"
        d_oldpeak, d_slope = 2.5, "Flat"
    else:
        d_age, d_sex, d_cp = 52, "M", "ASY"
        d_bp, d_chol, d_fbs = 130, 225, "Normal (<= 120 mg/dl)"
        d_ecg, d_maxhr, d_exang = "Normal", 145, "No"
        d_oldpeak, d_slope = 0.8, "Flat"

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Demographics & Symptoms")
            age = st.slider("Patient Age (Years)", min_value=20, max_value=85, value=d_age, step=1)
            
            sex = st.selectbox("Sex", options=["M", "F"], index=0 if d_sex == "M" else 1,
                               format_func=lambda x: "Male (M)" if x == "M" else "Female (F)")
            
            cp_options = ["ASY", "NAP", "ATA", "TA"]
            cp_names = {
                "ASY": "Asymptomatic (ASY)",
                "NAP": "Non-Anginal Pain (NAP)",
                "ATA": "Atypical Angina (ATA)",
                "TA": "Typical Angina (TA)"
            }
            chest_pain = st.selectbox(
                "Chest Pain Type",
                options=cp_options,
                index=cp_options.index(d_cp),
                format_func=lambda x: cp_names[x],
                help="ASY: Asymptomatic; NAP: Non-Anginal; ATA: Atypical Angina; TA: Typical Angina"
            )
            
            ex_angina = st.radio(
                "Exercise Induced Angina",
                options=["N", "Y"],
                index=0 if d_exang == "No" or d_exang == "N" else 1,
                format_func=lambda x: "No (N)" if x == "N" else "Yes (Y)",
                horizontal=True,
                help="Does the patient experience chest pain during physical exercise?"
            )

        with col2:
            st.markdown("#### 🩺 Vitals & Laboratory")
            resting_bp = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=70, max_value=220, value=d_bp, step=1,
                help="Normal systolic pressure is generally < 120 mm Hg."
            )
            
            cholesterol = st.number_input(
                "Serum Cholesterol (mg/dL)",
                min_value=80, max_value=600, value=d_chol, step=1,
                help="Desirable cholesterol level is generally < 200 mg/dL."
            )
            
            fbs_choice = st.selectbox(
                "Fasting Blood Sugar",
                options=["Normal (<= 120 mg/dl)", "High (> 120 mg/dl)"],
                index=0 if "Normal" in d_fbs else 1,
                help="Indicator of diabetes or impaired fasting glucose (> 120 mg/dL)"
            )
            fasting_bs = 1 if "High" in fbs_choice else 0
            
            max_hr = st.slider(
                "Max Heart Rate Achieved (MaxHR)",
                min_value=60, max_value=220, value=d_maxhr, step=1,
                help="Highest heart rate recorded during peak exercise treadmill test."
            )

        with col3:
            st.markdown("#### ⚡ Electrocardiogram (ECG)")
            ecg_options = ["Normal", "ST", "LVH"]
            ecg_labels = {
                "Normal": "Normal",
                "ST": "ST-T Wave Abnormality (ST)",
                "LVH": "Left Ventricular Hypertrophy (LVH)"
            }
            resting_ecg = st.selectbox(
                "Resting ECG Results",
                options=ecg_options,
                index=ecg_options.index(d_ecg),
                format_func=lambda x: ecg_labels[x],
                help="Resting electrocardiogram evaluation."
            )
            
            oldpeak = st.slider(
                "ST Depression (Oldpeak)",
                min_value=-2.0, max_value=6.0, value=float(d_oldpeak), step=0.1,
                help="ST depression induced by exercise relative to resting state."
            )
            
            slope_options = ["Up", "Flat", "Down"]
            slope_labels = {
                "Up": "Upsloping (Up)",
                "Flat": "Flat (Flat)",
                "Down": "Downsloping (Down)"
            }
            st_slope = st.selectbox(
                "Peak Exercise ST Segment Slope",
                options=slope_options,
                index=slope_options.index(d_slope),
                format_func=lambda x: slope_labels[x],
                help="Slope of the peak exercise ST segment. 'Flat' and 'Down' indicate ischemia."
            )
            
        submitted = st.form_submit_button("🔍 Run Heart Disease Risk Assessment", use_container_width=True, type="primary")

    if submitted or preset != "Default / Custom":
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'Age': age,
            'Sex': sex,
            'ChestPainType': chest_pain,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'RestingECG': resting_ecg,
            'MaxHR': max_hr,
            'ExerciseAngina': ex_angina,
            'Oldpeak': oldpeak,
            'ST_Slope': st_slope
        }])
        
        # Predict using selected pipeline
        prediction = active_pipeline.predict(input_data)[0]
        prediction_proba = active_pipeline.predict_proba(input_data)[0]
        prob_heart_disease = prediction_proba[1]
        
        st.divider()
        st.subheader("📋 Assessment Results")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            # Plotly Gauge Chart for Risk Probability
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_heart_disease * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Heart Disease Risk ({selected_model_name})", 'font': {'size': 20}},
                number={'suffix': "%", 'font': {'size': 32}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#d32f2f" if prob_heart_disease > 0.5 else "#388e3c", 'thickness': 0.25},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#ccc",
                    'steps': [
                        {'range': [0, 35], 'color': "rgba(76, 175, 80, 0.25)"},
                        {'range': [35, 65], 'color': "rgba(255, 193, 7, 0.25)"},
                        {'range': [65, 100], 'color': "rgba(244, 67, 54, 0.25)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if prob_heart_disease >= 0.65:
                st.markdown("""
                <div class="risk-badge-high" style="font-size: 1.1rem; width: 100%; text-align: center;">
                    ⚠️ HIGH RISK: Heart Disease Detected ({:.1f}%)
                </div>
                """.format(prob_heart_disease * 100), unsafe_allow_html=True)
            elif prob_heart_disease >= 0.35:
                st.markdown("""
                <div class="risk-badge-medium" style="font-size: 1.1rem; width: 100%; text-align: center;">
                    ⚡ MODERATE / BORDERLINE RISK ({:.1f}%)
                </div>
                """.format(prob_heart_disease * 100), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="risk-badge-low" style="font-size: 1.1rem; width: 100%; text-align: center;">
                    ✅ LOW RISK: Normal Cardiac Profile ({:.1f}%)
                </div>
                """.format(prob_heart_disease * 100), unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### 🔍 Identified Risk Factors & Clinical Indicators")
            
            identified_risks = []
            if st_slope in ["Flat", "Down"]:
                identified_risks.append(("ST Slope Abnormality", f"ST Slope is '{st_slope}', strongly associated with myocardial ischemia.", "🔴"))
            if oldpeak > 1.0:
                identified_risks.append(("Significant ST Depression", f"Oldpeak value of {oldpeak} mm indicates exercise-induced myocardial stress.", "🔴"))
            if ex_angina == "Y":
                identified_risks.append(("Exercise-Induced Angina", "Positive for chest discomfort during physical exertion.", "🔴"))
            if chest_pain == "ASY":
                identified_risks.append(("Asymptomatic Presentation", "Asymptomatic cases often mask silent coronary artery disease.", "🟠"))
            if resting_bp >= 140:
                identified_risks.append(("Stage 2 Hypertension", f"Resting blood pressure ({resting_bp} mmHg) is elevated.", "🟠"))
            if cholesterol >= 240:
                identified_risks.append(("Hypercholesterolemia", f"Serum cholesterol ({cholesterol} mg/dL) is high (desirable < 200).", "🟠"))
            if fasting_bs == 1:
                identified_risks.append(("Elevated Fasting Glucose", "Fasting blood sugar > 120 mg/dL increases vascular risk.", "🟠"))
            if max_hr < 120 and age < 65:
                identified_risks.append(("Chronotropic Incompetence", f"Max heart rate ({max_hr} bpm) is blunted for age {age}.", "🟡"))

            if identified_risks:
                for title, desc, icon in identified_risks:
                    st.markdown(f"**{icon} {title}:** {desc}")
            else:
                st.success("🟢 No elevated cardiovascular biomarker alerts triggered in this evaluation.")

            st.markdown("---")
            st.markdown("#### 💡 Clinical Guidance & Recommendations")
            if prob_heart_disease >= 0.5:
                st.warning("""
                - **Immediate Step:** Recommend comprehensive diagnostic evaluation (Coronary Angiography or Stress Echocardiography).
                - **Monitoring:** Daily blood pressure tracking and lipid panel re-evaluation.
                - **Lifestyle:** Supervised cardiovascular rehabilitation program and low-sodium Mediterranean diet.
                """)
            else:
                st.info("""
                - **Preventive Care:** Routine annual cardiology and metabolic health check-ups.
                - **Target Metrics:** Maintain systolic BP < 120 mmHg, LDL < 100 mg/dL, and regular 150 min/week moderate aerobic activity.
                """)

# ==============================================================================
# TAB 2: MODEL BENCHMARKS & COMPARISON
# ==============================================================================
with tab2:
    st.subheader("Comparative Machine Learning Benchmark")
    st.markdown("We trained and rigorously cross-validated 5 different machine learning classifiers on the heart disease cohort.")
    
    # Summary Table
    comparison_df = pd.DataFrame([
        {
            'Algorithm': model_name,
            'Test Accuracy': f"{m['accuracy'] * 100:.2f}%",
            'Precision': f"{m['precision'] * 100:.2f}%",
            'Recall (Sensitivity)': f"{m['recall'] * 100:.2f}%",
            'F1 Score': f"{m['f1_score']:.4f}",
            'ROC-AUC': f"{m['roc_auc']:.4f}",
            '5-Fold CV Acc': f"{m['cv_mean_acc'] * 100:.2f}% (±{m['cv_std_acc'] * 100:.2f}%)"
        }
        for model_name, m in metrics.items()
    ])
    st.dataframe(comparison_df.style.highlight_max(subset=['F1 Score', 'ROC-AUC'], color='#d1e7dd'), use_container_width=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Metrics Bar Chart
        metric_plot_data = []
        for name, m in metrics.items():
            metric_plot_data.extend([
                {'Model': name, 'Metric': 'Accuracy', 'Score': m['accuracy']},
                {'Model': name, 'Metric': 'F1 Score', 'Score': m['f1_score']},
                {'Model': name, 'Metric': 'ROC-AUC', 'Score': m['roc_auc']}
            ])
        fig_metrics = px.bar(
            pd.DataFrame(metric_plot_data),
            x='Model', y='Score', color='Metric', barmode='group',
            title="Model Evaluation Metrics Comparison",
            color_discrete_sequence=['#4285F4', '#EA4335', '#FBBC05']
        )
        fig_metrics.update_layout(yaxis_range=[0.75, 1.0], height=380)
        st.plotly_chart(fig_metrics, use_container_width=True)

    with col_chart2:
        # Feature Importance Chart
        feat_df = pd.DataFrame(top_features[:10])
        fig_feat = px.bar(
            feat_df,
            x='importance', y='feature',
            orientation='h',
            title="Top 10 Feature Importances (Random Forest)",
            color='importance',
            color_continuous_scale='Reds'
        )
        fig_feat.update_layout(yaxis=dict(autorange="reversed"), height=380)
        st.plotly_chart(fig_feat, use_container_width=True)
        
    st.divider()
    st.subheader("Confusion Matrix Inspector")
    cm_model = st.selectbox("Select Model for Confusion Matrix:", list(metrics.keys()), index=0)
    cm = np.array(metrics[cm_model]['confusion_matrix'])
    
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=['No Disease (0)', 'Heart Disease (1)'],
        y=['No Disease (0)', 'Heart Disease (1)'],
        color_continuous_scale='Blues',
        title=f"Confusion Matrix for {cm_model}"
    )
    cm_fig.update_layout(height=340, width=500)
    st.plotly_chart(cm_fig, use_container_width=True)

# ==============================================================================
# TAB 3: DATASET ANALYTICS & EDA
# ==============================================================================
with tab3:
    st.subheader("Exploratory Data Analysis (EDA)")
    st.markdown("Insights and biomarker distributions from the 918-patient Heart Disease dataset.")
    
    if df_dataset is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Patients", len(df_dataset))
        c2.metric("Heart Disease Cases", f"{int(df_dataset['HeartDisease'].sum())} ({df_dataset['HeartDisease'].mean() * 100:.1f}%)")
        c3.metric("Normal Patients", f"{int((df_dataset['HeartDisease'] == 0).sum())} ({(1 - df_dataset['HeartDisease'].mean()) * 100:.1f}%)")
        c4.metric("Avg Patient Age", f"{df_dataset['Age'].mean():.1f} yrs")
        
        eda_col1, eda_col2 = st.columns(2)
        with eda_col1:
            fig_age = px.histogram(
                df_dataset, x="Age", color="HeartDisease",
                barmode="overlay", nbins=25,
                title="Age Distribution by Heart Disease Status",
                color_discrete_map={0: "#4CAF50", 1: "#E53935"},
                labels={"HeartDisease": "Disease Status"}
            )
            st.plotly_chart(fig_age, use_container_width=True)
            
            fig_cp = px.histogram(
                df_dataset, x="ChestPainType", color="HeartDisease",
                barmode="group",
                title="Chest Pain Type vs Heart Disease Frequency",
                color_discrete_map={0: "#4CAF50", 1: "#E53935"}
            )
            st.plotly_chart(fig_cp, use_container_width=True)

        with eda_col2:
            fig_chol = px.box(
                df_dataset[df_dataset['Cholesterol'] > 0],
                x="HeartDisease", y="Cholesterol", color="HeartDisease",
                title="Cholesterol Levels by Heart Disease Outcome",
                color_discrete_map={0: "#4CAF50", 1: "#E53935"}
            )
            st.plotly_chart(fig_chol, use_container_width=True)
            
            fig_slope = px.histogram(
                df_dataset, x="ST_Slope", color="HeartDisease",
                barmode="group",
                title="ST Slope Distribution vs Heart Disease Outcome",
                color_discrete_map={0: "#4CAF50", 1: "#E53935"}
            )
            st.plotly_chart(fig_slope, use_container_width=True)

        st.subheader("Raw Dataset Explorer")
        st.dataframe(df_dataset.head(50), use_container_width=True)
    else:
        st.warning("heart.csv dataset file not found in directory.")

# ==============================================================================
# TAB 4: BATCH PREDICTION
# ==============================================================================
with tab4:
    st.subheader("Batch Patient Evaluation via CSV")
    st.markdown("Upload a CSV file containing patient metrics to perform bulk heart disease classification.")
    
    col_download_template, col_uploader = st.columns([1, 2])
    
    with col_download_template:
        st.markdown("##### 📄 Need a Template?")
        if df_dataset is not None:
            sample_template = df_dataset.drop(columns=['HeartDisease']).head(5)
            csv_sample = sample_template.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Sample CSV Template",
                data=csv_sample,
                file_name="heart_disease_batch_sample.csv",
                mime="text/csv",
                use_container_width=True
            )
        st.caption("Required columns: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope")

    with col_uploader:
        uploaded_file = st.file_uploader("Upload Batch CSV", type=["csv"])
        
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} patient records.")
            
            # Check required columns
            required_cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
            missing = [c for c in required_cols if c not in batch_df.columns]
            
            if missing:
                st.error(f"Missing required columns in CSV: {missing}")
            else:
                preds = active_pipeline.predict(batch_df[required_cols])
                probas = active_pipeline.predict_proba(batch_df[required_cols])[:, 1]
                
                result_df = batch_df.copy()
                result_df['Predicted_HeartDisease'] = preds
                result_df['Disease_Probability_%'] = (probas * 100).round(2)
                result_df['Risk_Level'] = [
                    'High Risk' if p >= 0.65 else ('Moderate Risk' if p >= 0.35 else 'Low Risk')
                    for p in probas
                ]
                
                st.success("Batch Inference Completed Successfully!")
                
                b1, b2, b3 = st.columns(3)
                b1.metric("High Risk Patients", int((result_df['Risk_Level'] == 'High Risk').sum()))
                b2.metric("Moderate Risk Patients", int((result_df['Risk_Level'] == 'Moderate Risk').sum()))
                b3.metric("Low Risk Patients", int((result_df['Risk_Level'] == 'Low Risk').sum()))
                
                st.dataframe(result_df, use_container_width=True)
                
                output_csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Annotated Predictions CSV",
                    data=output_csv,
                    file_name="cardiac_predictions_output.csv",
                    mime="text/csv",
                    type="primary"
                )
        except Exception as e:
            st.error(f"Error processing batch CSV: {str(e)}")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #757575; font-size: 0.85rem;">
    ⚠️ <b>Medical Disclaimer:</b> This tool is built strictly for educational, demonstration, and clinical decision-support research purposes. It does not replace professional medical judgment or physician diagnosis.
</div>
""", unsafe_allow_html=True)
