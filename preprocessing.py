import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")

print(df.head())

print("\nShape:")
print(df.shape)


# ==========================================
# 2. HANDLE MISSING VALUES
# ==========================================

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill BMI missing values with median
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# Fill smoking status missing values
df["smoking_status"] = df["smoking_status"].fillna("Unknown")

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ==========================================
# 3. REMOVE ID
# ==========================================

df = df.drop("id", axis=1)

print("\nAfter removing ID:")
print(df.head())

print("\nColumns after removing ID:")
print(df.columns)


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("stroke", axis=1)

y = df["stroke"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nShape of X:")
print(X.shape)

print("\nShape of y:")
print(y.shape)


# ==========================================
# 5. ENCODE CATEGORICAL COLUMNS
# ==========================================

X = pd.get_dummies(
    X,
    drop_first=True
)

# Convert all columns to float
X = X.astype(float)

print("\nAfter encoding categorical columns:")
print(X.head())

print("\nShape of X after encoding:")
print(X.shape)

print("\nData types:")
print(X.dtypes)


# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ==========================================
# 7. APPLY SMOTE
# ==========================================

print("\nBefore SMOTE:")
print(y_train.value_counts())

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

print("\nResampled training data shape:")
print(X_train_smote.shape)


# ==========================================
# 8. RANDOM FOREST
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train_smote,
    y_train_smote
)

print("\nRandom Forest model trained successfully!")


# Predictions
rf_prediction = rf_model.predict(X_test)

rf_probability = rf_model.predict_proba(X_test)[:, 1]


# Accuracy
rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)

print("\nRandom Forest Accuracy:")
print(f"{rf_accuracy * 100:.2f} %")


# Classification Report
print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        rf_prediction
    )
)


# Confusion Matrix
rf_cm = confusion_matrix(
    y_test,
    rf_prediction
)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)


# ROC-AUC
rf_auc = roc_auc_score(
    y_test,
    rf_probability
)

print("\nRandom Forest ROC-AUC:")
print(f"{rf_auc:.4f}")


# ==========================================
# 9. SCALE DATA FOR LOGISTIC REGRESSION
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_smote
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================
# 10. LOGISTIC REGRESSION
# ==========================================

logistic_model = LogisticRegression(
    max_iter=5000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train_smote
)

print("\nLogistic Regression model trained successfully!")


# Predictions
lr_prediction = logistic_model.predict(
    X_test_scaled
)

lr_probability = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]


# Accuracy
lr_accuracy = accuracy_score(
    y_test,
    lr_prediction
)

print("\nLogistic Regression Accuracy:")
print(f"{lr_accuracy * 100:.2f} %")


# Classification Report
print("\nLogistic Regression Classification Report:")
print(
    classification_report(
        y_test,
        lr_prediction
    )
)


# Confusion Matrix
lr_cm = confusion_matrix(
    y_test,
    lr_prediction
)

print("\nLogistic Regression Confusion Matrix:")
print(lr_cm)


# ROC-AUC
lr_auc = roc_auc_score(
    y_test,
    lr_probability
)

print("\nLogistic Regression ROC-AUC:")
print(f"{lr_auc:.4f}")


# ==========================================
# 11. PRECISION-RECALL SCORE
# ==========================================

lr_average_precision = average_precision_score(
    y_test,
    lr_probability
)

print("\nLogistic Regression Average Precision:")
print(f"{lr_average_precision:.4f}")


# ==========================================
# 12. MODEL COMPARISON
# ==========================================

print("\n")
print("==========================================")
print("             MODEL COMPARISON")
print("==========================================")

print(
    f"\nRandom Forest Accuracy: "
    f"{rf_accuracy * 100:.2f} %"
)

print(
    f"Random Forest ROC-AUC: "
    f"{rf_auc:.4f}"
)

print(
    f"\nLogistic Regression Accuracy: "
    f"{lr_accuracy * 100:.2f} %"
)

print(
    f"Logistic Regression ROC-AUC: "
    f"{lr_auc:.4f}"
)

print(
    f"Logistic Regression Average Precision: "
    f"{lr_average_precision:.4f}"
)


# ==========================================
# 13. ROC CURVE
# ==========================================

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probability
)

lr_fpr, lr_tpr, _ = roc_curve(
    y_test,
    lr_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_auc:.2f})"
)

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC = {lr_auc:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()


# ==========================================
# 14. PRECISION-RECALL CURVE
# ==========================================

lr_precision, lr_recall, _ = precision_recall_curve(
    y_test,
    lr_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    lr_recall,
    lr_precision,
    label=f"Logistic Regression (AP = {lr_average_precision:.2f})"
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.legend()

plt.show()


# ==========================================
# 15. SAVE MODEL
# ==========================================

joblib.dump(
    logistic_model,
    "stroke_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "features.pkl"
)

print("\nModel, scaler, and features saved successfully!")


# ==========================================
# 16. COMPLETION MESSAGE
# ==========================================

print("\n")
print("==========================================")
print(" PREPROCESSING AND MODEL EVALUATION")
print("          COMPLETED SUCCESSFULLY!")
print("==========================================")