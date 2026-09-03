import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "data/bongo_scam.csv"

df = pd.read_csv(DATASET_PATH)

print("\n===== DATASET =====")
print(f"Original messages: {len(df)}")


# ============================================================
# 2. REMOVE DUPLICATES
# ============================================================

original_size = len(df)

df = df.drop_duplicates(subset=["Sms"]).copy()

duplicates_removed = original_size - len(df)

print(f"Duplicate messages removed: {duplicates_removed}")
print(f"Messages remaining: {len(df)}")


# ============================================================
# 3. ENCODE LABELS
# ============================================================

df["Category"] = df["Category"].map({
    "trust": 0,
    "scam": 1
})

# Make sure every label was successfully converted
if df["Category"].isnull().any():
    raise ValueError(
        "Some Category values could not be converted. "
        "Expected labels: 'trust' and 'scam'."
    )

print("\n===== CLASS DISTRIBUTION =====")
print(df["Category"].value_counts())

print("\nClass percentages:")
print(
    (df["Category"].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# 4. SEPARATE INPUTS AND LABELS
# ============================================================

X = df["Sms"]
y = df["Category"]


# ============================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First split:
# 70% training
# 30% temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split:
# Half of the remaining 30% → validation
# Half of the remaining 30% → test

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\n===== DATA SPLIT =====")
print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples:       {len(X_test)}")


# ============================================================
# 6. TF-IDF VECTORIZATION
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=10000
)

# Learn the vocabulary and IDF values ONLY from training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Apply the learned vocabulary to validation and test
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print("\n===== TF-IDF =====")
print(f"Training features:   {X_train_tfidf.shape}")
print(f"Validation features: {X_val_tfidf.shape}")
print(f"Test features:       {X_test_tfidf.shape}")
print(f"Vocabulary size:     {len(vectorizer.vocabulary_)}")


# ============================================================
# 7. CREATE XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\n===== TRAINING =====")
print("Training XGBoost model...")

model.fit(
    X_train_tfidf,
    y_train
)

print("Training completed.")


# ============================================================
# 9. VALIDATION EVALUATION
# ============================================================

print("\n===== VALIDATION =====")

y_val_pred = model.predict(X_val_tfidf)
y_val_prob = model.predict_proba(X_val_tfidf)[:, 1]

val_accuracy = accuracy_score(
    y_val,
    y_val_pred
)

val_auc = roc_auc_score(
    y_val,
    y_val_prob
)

print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Validation ROC-AUC:  {val_auc:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_val,
        y_val_pred,
        target_names=["Trust", "Scam"]
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        y_val,
        y_val_pred
    )
)


# ============================================================
# 10. FINAL TEST EVALUATION
# ============================================================

print("\n===== FINAL TEST =====")

# IMPORTANT:
# The test set has not been used to train or tune the model.

y_test_pred = model.predict(X_test_tfidf)
y_test_prob = model.predict_proba(X_test_tfidf)[:, 1]

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

test_auc = roc_auc_score(
    y_test,
    y_test_prob
)

test_cm = confusion_matrix(
    y_test,
    y_test_pred
)

print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

print(f"Test ROC-AUC:  {test_auc:.4f}")

print("\nFinal classification report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["Trust", "Scam"]
    )
)

print("Final confusion matrix:")
print(test_cm)


# ============================================================
# 11. SAVE TRAINED MODEL
# ============================================================

MODEL_PATH = "sms_fraud_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

print("\n===== MODEL SAVED =====")
print(f"Model:      {MODEL_PATH}")
print(f"Vectorizer: {VECTORIZER_PATH}")


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Dataset size:           {len(df)}")
print(f"Training samples:      {len(X_train)}")
print(f"Validation samples:    {len(X_val)}")
print(f"Test samples:          {len(X_test)}")

print(f"\nValidation accuracy:   {val_accuracy * 100:.2f}%")
print(f"Validation ROC-AUC:    {val_auc:.4f}")

print(f"\nFinal test accuracy:   {test_accuracy * 100:.2f}%")
print(f"Final test ROC-AUC:    {test_auc:.4f}")

print("\nSaved artifacts:")
print(f"  {MODEL_PATH}")
print(f"  {VECTORIZER_PATH}")

print("\nThe trained model is ready for integration.")