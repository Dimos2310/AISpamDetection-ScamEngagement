import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def main():
    # Load Q2 preprocessed dataset
    dataset_path = Path("Q2/training_dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError("Q2/training_dataset.csv not found. Please run preprocess_question2.py first.")

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    # Features list
    feature_cols = [
        "char_count", "word_count", "sentence_count", "paragraph_count", "line_count",
        "avg_word_length", "avg_sentence_length", "uppercase_ratio", "digit_ratio",
        "punctuation_count", "exclamation_count", "question_count", "multiple_exclamation_present",
        "all_caps_word_count", "url_count", "email_count", "phone_count", "money_expression_count",
        "anonymized_entity_token_count", "urgency_count", "secrecy_count", "financial_count",
        "bank_count", "personal_info_count", "action_request_count", "fee_count", "greeting_count",
        "signature_count", "title_count", "country_count", "organization_count", "starts_with_greeting",
        "ends_with_signature", "has_urgent_action_pattern", "has_confidentiality_pattern",
        "has_large_money_pattern", "has_bank_details_request_pattern", "has_personal_info_request_pattern",
        "has_advance_fee_pattern", "body_is_empty"
    ]

    X = df[feature_cols]
    y = df['flag_binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Fitting feature scaler (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    print("Training Support Vector Machine (SVM) Model with RBF Kernel...")
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)

    # Evaluate SVM
    y_pred_svm = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred_svm)
    print("\n" + "="*40)
    print(f"SVM Test Accuracy: {accuracy:.4%}")
    print("="*40)
    print("Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=["Legitimate", "Spam/Scam"]))
    print("="*40)

    # Save model and scaler
    model_path = Path("spam_model.pkl")
    scaler_path = Path("scaler.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(svm, f)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Successfully saved trained classifier to: {model_path}")
    print(f"Successfully saved feature scaler to:    {scaler_path}")

if __name__ == "__main__":
    main()
