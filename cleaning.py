import pandas as pd

df = pd.read_csv("dataset.csv")

print("Before cleaning:")
print(df.isnull().sum())

# Fill missing BMI values with median
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

print("\nAfter cleaning BMI:")
print(df["bmi"].isnull().sum())

# Fill missing smoking status with "Unknown"
df["smoking_status"] = df["smoking_status"].fillna("Unknown")

print("\nAfter cleaning smoking_status:")
print(df["smoking_status"].isnull().sum())

print("\nFinal Missing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nUnique values in categorical columns:")

print("\nGender:")
print(df["gender"].unique())

print("\nEver Married:")
print(df["ever_married"].unique())

print("\nWork Type:")
print(df["work_type"].unique())

print("\nResidence Type:")
print(df["Residence_type"].unique())

print("\nSmoking Status:")
print(df["smoking_status"].unique())

print("\nStroke:")
print(df["stroke"].unique())

# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")




