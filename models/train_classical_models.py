import os
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

from utils.preprocess import preprocess_text, get_vectorizer

def main():
    """Train classical ML models for Latvian clickbait detection."""
    print("="*60)
    print("Training Classical Models for Clickbait Detection")
    print("="*60)
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "klikšķēsma.txt")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n1. Loading data...")
    df = pd.read_csv(
        data_path,
        sep="\t",
        names=["title", "label"],
        encoding="utf-8"
    )
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts().sort_index()}")
    
    assert len(df) == 4930, f"Expected 4930 samples, got {len(df)}"
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["title"],
        df["label"],
        test_size=0.3,
        random_state=42,
        stratify=df["label"]
    )
    
    print(f"\nTrain samples: {len(X_train_raw)}")
    print(f"Test samples: {len(X_test_raw)}")
    
    print("\n2. Preprocessing texts...")
    X_train_clean = X_train_raw.apply(
        lambda t: preprocess_text(
            t, 
            casefold=False, 
            morpho_method="lemmatize",
            keep_numbers=True,
            keep_exclamation=True
        )
    )
    X_test_clean = X_test_raw.apply(
        lambda t: preprocess_text(
            t, 
            casefold=False, 
            morpho_method="lemmatize",
            keep_numbers=True,
            keep_exclamation=True
        )
    )
    
    print("\n3. Vectorizing with TF-IDF...")
    vectorizer = get_vectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_clean)
    X_test = vectorizer.transform(X_test_clean)
    
    print(f"Train matrix shape: {X_train.shape}")
    print(f"Test matrix shape: {X_test.shape}")
    
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved vectorizer to: {vectorizer_path}")
    
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "naive_bayes": MultinomialNB(),
        "svm": LinearSVC(random_state=42, max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5)
    }
    
    print("\n4. Training and evaluating models...")
    print("-" * 80)
    header = f"{'Model':<25}  {'Acc':<6} {'Prec':<6} {'Rec':<6} {'F1':<6} {'ROC-AUC':<8}"
    print(header)
    print("-" * 80)
    
    results = {}
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X_test)
            if len(y_score.shape) == 1:
                y_score = np.column_stack([-y_score, y_score])
            y_score = np.exp(y_score) / np.sum(np.exp(y_score), axis=1, keepdims=True)
        else:
            y_score = None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro")
        
        if y_score is not None:
            roc = roc_auc_score(y_test, y_score, multi_class="ovo", average="macro")
        else:
            roc = np.nan
        
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc
        }
        
        print(f"{name:<25}  {acc:<6.3f} {prec:<6.3f} {rec:<6.3f} {f1:<6.3f} {roc:<8.3f}")
        
        model_path = os.path.join(models_dir, f"{name}.joblib")
        joblib.dump(clf, model_path)
        print(f"  → Saved to: {model_path}")
        
        print(f"\nDetailed classification report for {name}:")
        print(classification_report(y_test, y_pred, target_names=['1-Nav', '2-Daļēja', '3-Ir']))
        print("-" * 80)
    
    results_df = pd.DataFrame(results).T
    results_path = os.path.join(models_dir, "classical_models_results.csv")
    results_df.to_csv(results_path)
    print(f"\nSaved results summary to: {results_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Models trained: {len(classifiers)}")
    print(f"Best F1 score: {results_df['f1'].max():.3f} ({results_df['f1'].idxmax()})")
    print(f"Best accuracy: {results_df['accuracy'].max():.3f} ({results_df['accuracy'].idxmax()})")
    
    metadata = {
        'training_date': datetime.now().isoformat(),
        'n_train_samples': len(X_train_raw),
        'n_test_samples': len(X_test_raw),
        'n_features': X_train.shape[1],
        'preprocessing': {
            'casefold': False,
            'morpho_method': 'lemmatize',
            'keep_numbers': True,
            'keep_exclamation': True
        },
        'vectorizer_params': {
            'max_features': 1000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'sublinear_tf': True
        }
    }
    
    import json
    metadata_path = os.path.join(models_dir, "training_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nSaved training metadata to: {metadata_path}")

if __name__ == "__main__":
    main()