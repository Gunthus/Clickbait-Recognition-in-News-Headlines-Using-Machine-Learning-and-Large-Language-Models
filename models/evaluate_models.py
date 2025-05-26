import os
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

from utils.preprocess import preprocess_text
from llm.gpt_zero_shot_classifier import classify_zero_shot_batch
from llm.mistral_classifier import classify_mistral_batch

def main():
    """Evaluate all models (classical and LLM) on the same test set."""
    print("="*60)
    print("EVALUATING ALL MODELS FOR CLICKBAIT DETECTION")
    print("="*60)
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "klikšķēsma.txt")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n1. Loading data...")
    df = pd.read_csv(
        data_path,
        sep="\t",
        names=["title", "label"],
        encoding="utf-8"
    )
    
    print(f"Total samples: {len(df)}")
    assert len(df) == 4930, f"Expected 4930 samples, got {len(df)}"
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["title"], df["label"],
        test_size=0.3,
        random_state=42,
        stratify=df["label"]
    )
    
    print(f"Test samples: {len(X_test_raw)} (should be 1479)")
    assert len(X_test_raw) == 1479, f"Expected 1479 test samples, got {len(X_test_raw)}"
    
    y_test_array = y_test.values

    print("\n2. Evaluating Classical Models...")
    print("-" * 60)
    
    vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    
    X_test_clean = X_test_raw.apply(
        lambda t: preprocess_text(
            t, 
            casefold=False, 
            morpho_method="lemmatize",
            keep_numbers=True,
            keep_exclamation=True
        )
    )
    X_test_vec = vectorizer.transform(X_test_clean)

    model_names = [
        "logistic_regression",
        "naive_bayes",
        "svm",
        "random_forest",
        "knn"
    ]
    
    classifiers = {}
    for name in model_names:
        model_path = os.path.join(models_dir, f"{name}.joblib")
        if os.path.exists(model_path):
            classifiers[name] = joblib.load(model_path)
            print(f"Loaded: {name}")
        else:
            print(f"Warning: Model file not found: {model_path}")

    print("\n" + "="*80)
    print("CLASSICAL MODELS PERFORMANCE")
    print("="*80)
    header = f"{'Model':<20}  {'Acc':<6} {'Prec':<6} {'Rec':<6} {'F1':<6} {'ROC-AUC':<8}"
    print(header)
    print("-" * 80)
    
    classical_results = {}
    
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test_vec)
        
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test_vec)
        elif hasattr(clf, "decision_function"):