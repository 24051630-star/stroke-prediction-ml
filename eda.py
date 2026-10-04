# =============================
# STROKE PREDICTION - EDA
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. LOAD DATASET
# -----------------------------

df = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")

# -----------------------------
# 2. FIRST 5 ROWS
# -----------------------------

print("\nFirst 5 Rows:")
print(df.head())

# -----------------------------
# 3. SHAPE OF DATASET
# -----------------------------

print("\nShape of Dataset:")
print(df.shape)

# -----------------------------
# 4. COLUMN NAMES
# -----------------------------

print("\nColumns:")
print(df.columns)

# -----------------------------
# 5. DATA TYPES
# -----------------------------

print("\nData Types:")
print(df.dtypes)

# -----------------------------
# 6. STATISTICAL SUMMARY
# -----------------------------

print("\nStatistical Summary:")
print(df.describe())

# -----------------------------
# 7. MISSING VALUES
# -----------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# 8. DUPLICATE ROWS
# -----------------------------

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# -----------------------------
# 9. STROKE DISTRIBUTION
# -----------------------------

print("\nStroke Distribution:")
print(df["stroke"].value_counts())


# =============================
# EDA VISUALIZATIONS
# =============================

# 10. Stroke Distribution

plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="stroke")

plt.title("Stroke Distribution")
plt.xlabel("Stroke (0 = No Stroke, 1 = Stroke)")
plt.ylabel("Number of Patients")

plt.show()


# 11. Age Distribution

plt.figure(figsize=(7, 4))

sns.histplot(data=df, x="age", bins=30, kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

plt.show()


# 12. Age vs Stroke

plt.figure(figsize=(7, 5))

sns.boxplot(data=df, x="stroke", y="age")

plt.title("Age vs Stroke")
plt.xlabel("Stroke (0 = No Stroke, 1 = Stroke)")
plt.ylabel("Age")

plt.show()


# 13. BMI vs Stroke

plt.figure(figsize=(7, 5))

sns.boxplot(data=df, x="stroke", y="bmi")

plt.title("BMI vs Stroke")
plt.xlabel("Stroke (0 = No Stroke, 1 = Stroke)")
plt.ylabel("BMI")

plt.show()


# 14. Average Glucose Level vs Stroke

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="stroke",
    y="avg_glucose_level"
)

plt.title("Average Glucose Level vs Stroke")
plt.xlabel("Stroke (0 = No Stroke, 1 = Stroke)")
plt.ylabel("Average Glucose Level")

plt.show()


# 15. Gender vs Stroke

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="gender",
    hue="stroke"
)

plt.title("Gender vs Stroke")
plt.xlabel("Gender")
plt.ylabel("Number of Patients")

plt.show()


# 16. Hypertension vs Stroke

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="hypertension",
    hue="stroke"
)

plt.title("Hypertension vs Stroke")
plt.xlabel("Hypertension (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")

plt.show()


# 17. Heart Disease vs Stroke

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="heart_disease",
    hue="stroke"
)

plt.title("Heart Disease vs Stroke")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")

plt.show()


# 18. Smoking Status vs Stroke

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="smoking_status",
    hue="stroke"
)

plt.title("Smoking Status vs Stroke")
plt.xlabel("Smoking Status")
plt.ylabel("Number of Patients")

plt.xticks(rotation=30)

plt.show()


# 19. Work Type vs Stroke

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="work_type",
    hue="stroke"
)

plt.title("Work Type vs Stroke")
plt.xlabel("Work Type")
plt.ylabel("Number of Patients")

plt.xticks(rotation=30)

plt.show()


# 20. Residence Type vs Stroke

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Residence_type",
    hue="stroke"
)

plt.title("Residence Type vs Stroke")
plt.xlabel("Residence Type")
plt.ylabel("Number of Patients")

plt.show()


# 21. Correlation Heatmap

plt.figure(figsize=(10, 7))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

