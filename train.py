import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn

from data_preprocessing import process_video

def train_and_evaluate(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=True)
    
    rf_model = RandomForestClassifier(
        n_estimators= 165, 
        max_depth=18, 
        min_samples_split= 5, 
        min_samples_leaf= 2,    
        random_state=42,
        n_jobs=-1
    )

    with mlflow.start_run():
        rf_model.fit(X_train, y_train)
        mlflow.sklearn.log_model(rf_model, "random_forest_model")
        
        y_pred = rf_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Log metrics to MLFlow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact("confusion_matrix.txt", cm)
        mlflow.log_artifact("classification_report.txt", report)

        print("Độ chính xác:", accuracy)
        print("Ma trận nhầm lẫn:\n", cm)
        print("Báo cáo phân loại:\n", report)
