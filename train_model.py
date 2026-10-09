import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

def train_and_evaluate(data_path="heart.csv", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Preprocessing 0 values in Cholesterol and RestingBP (as observed in notebook)
    ch_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
    bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
    
    df['Cholesterol'] = df['Cholesterol'].replace(0, round(ch_mean, 2))
    df['RestingBP'] = df['RestingBP'].replace(0, round(bp_mean, 2))
    
    # Features and target
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']
    
    # Feature groups
    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    passthrough_features = ['FastingBS']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
            ('pass', 'passthrough', passthrough_features)
        ],
        remainder='drop'
    )
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Candidate models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=4, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=3, random_state=42),
        'Logistic Regression': LogisticRegression(C=0.5, max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7)
    }
    
    results = {}
    fitted_pipelines = {}
    best_model_name = None
    best_f1 = -1.0
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'roc_auc': round(float(auc), 4),
            'cv_mean_acc': round(float(cv_scores.mean()), 4),
            'cv_std_acc': round(float(cv_scores.std()), 4),
            'confusion_matrix': cm
        }
        
        fitted_pipelines[name] = pipeline
        print(f"[{name}] Acc: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f} | CV Acc: {cv_scores.mean():.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"\nBest Model: {best_model_name} (F1: {best_f1:.4f})")
    
    # Extract feature names from preprocessor
    fitted_preprocessor = fitted_pipelines[best_model_name].named_steps['preprocessor']
    cat_feature_names = fitted_preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names + passthrough_features
    
    # Extract feature importance for Random Forest
    rf_pipeline = fitted_pipelines['Random Forest']
    rf_importances = rf_pipeline.named_steps['classifier'].feature_importances_
    feat_importance_dict = sorted(
        [{'feature': feat, 'importance': round(float(imp), 4)} for feat, imp in zip(all_feature_names, rf_importances)],
        key=lambda x: x['importance'],
        reverse=True
    )
    
    # Save artifacts
    artifacts = {
        'best_model_name': best_model_name,
        'pipelines': fitted_pipelines,
        'imputation_values': {
            'Cholesterol_mean': round(ch_mean, 2),
            'RestingBP_mean': round(bp_mean, 2)
        },
        'feature_names': all_feature_names,
        'feature_importances': feat_importance_dict,
        'metrics': results,
        'categorical_options': {
            'Sex': sorted(df['Sex'].unique().tolist()),
            'ChestPainType': sorted(df['ChestPainType'].unique().tolist()),
            'RestingECG': sorted(df['RestingECG'].unique().tolist()),
            'ExerciseAngina': sorted(df['ExerciseAngina'].unique().tolist()),
            'ST_Slope': sorted(df['ST_Slope'].unique().tolist()),
            'FastingBS': [0, 1]
        },
        'numerical_ranges': {
            col: {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
            } for col in numeric_features
        }
    }
    
    model_save_path = os.path.join(output_dir, "heart_disease_artifacts.joblib")
    joblib.dump(artifacts, model_save_path)
    print(f"Model artifacts saved to {model_save_path}")
    
    # Save json summary for fast inspection
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            'best_model_name': best_model_name,
            'metrics': results,
            'top_features': feat_importance_dict[:8],
            'test_set_size': len(y_test),
            'train_set_size': len(y_train)
        }, f, indent=2)
    print(f"Metrics summary saved to {summary_path}")

if __name__ == '__main__':
    train_and_evaluate(
        data_path="/Users/chhaterpaldangi/.gemini/antigravity/scratch/heart_disease_prediction/heart.csv",
        output_dir="/Users/chhaterpaldangi/.gemini/antigravity/scratch/heart_disease_prediction/models"
    )
