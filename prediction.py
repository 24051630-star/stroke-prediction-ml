import pandas as pd
import joblib


# ==============================
# 1. Load saved files
# ==============================

model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")


print("\n================================")
print("      STROKE PREDICTION SYSTEM")
print("================================")


# ==============================
# 2. Helper functions
# ==============================

def get_gender():
    while True:
        value = input(
            "\nEnter gender (Male/Female/Other): "
        ).strip().lower()

        if value in ["male", "female", "other"]:
            return value.capitalize()

        print("Invalid input! Please enter Male, Female, or Other.")


def get_binary(question):
    while True:
        value = input(question).strip()

        if value in ["0", "1"]:
            return int(value)

        print("Invalid input! Please enter 0 or 1.")


def get_yes_no(question):
    while True:
        value = input(question).strip().lower()

        if value in ["yes", "no"]:
            return value.capitalize()

        print("Invalid input! Please enter Yes or No.")


def get_work_type():
    while True:
        value = input(
            "Work type "
            "(Private/Self-employed/Government/children/Never_worked): "
        ).strip().lower()

        work_types = {
            "private": "Private",
            "self-employed": "Self-employed",
            "government": "Government",
            "children": "children",
            "never_worked": "Never_worked",
            "never worked": "Never_worked"
        }

        if value in work_types:
            return work_types[value]

        print(
            "Invalid input! Please enter "
            "Private, Self-employed, Government, "
            "children, or Never_worked."
        )


def get_residence():
    while True:
        value = input(
            "Residence type (Urban/Rural): "
        ).strip().lower()

        if value in ["urban", "rural"]:
            return value.capitalize()

        print("Invalid input! Please enter Urban or Rural.")


def get_smoking_status():
    while True:
        value = input(
            "Smoking status "
            "(formerly smoked/never smoked/smokes/Unknown): "
        ).strip().lower()

        smoking_types = {
            "formerly smoked": "formerly smoked",
            "never smoked": "never smoked",
            "smokes": "smokes",
            "unknown": "Unknown"
        }

        if value in smoking_types:
            return smoking_types[value]

        print(
            "Invalid input! Please enter "
            "formerly smoked, never smoked, smokes, or Unknown."
        )


# ==============================
# 3. Take user input
# ==============================

gender = get_gender()


while True:
    try:
        age = float(input("Enter age: "))

        if age > 0:
            break

        print("Age must be greater than 0.")

    except ValueError:
        print("Please enter a valid number.")


hypertension = get_binary(
    "Do you have hypertension? (0 = No, 1 = Yes): "
)

heart_disease = get_binary(
    "Do you have heart disease? (0 = No, 1 = Yes): "
)

ever_married = get_yes_no(
    "Ever married? (Yes/No): "
)

work_type = get_work_type()

residence_type = get_residence()


while True:
    try:
        avg_glucose_level = float(
            input("Enter average glucose level: ")
        )

        if avg_glucose_level > 0:
            break

        print("Glucose level must be greater than 0.")

    except ValueError:
        print("Please enter a valid number.")


while True:
    try:
        bmi = float(input("Enter BMI: "))

        if bmi > 0:
            break

        print("BMI must be greater than 0.")

    except ValueError:
        print("Please enter a valid number.")


smoking_status = get_smoking_status()


# ==============================
# 4. Create DataFrame
# ==============================

input_data = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "ever_married": [ever_married],
    "work_type": [work_type],
    "Residence_type": [residence_type],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "smoking_status": [smoking_status]
})


# ==============================
# 5. Encode categorical columns
# ==============================

input_data = pd.get_dummies(
    input_data,
    drop_first=True
)


# ==============================
# 6. Match training features
# ==============================

input_data = input_data.reindex(
    columns=features,
    fill_value=0
)

input_data = input_data.astype(float)


# ==============================
# 7. Scale input
# ==============================

input_scaled = scaler.transform(input_data)


# ==============================
# 8. Make prediction
# ==============================

prediction = model.predict(input_scaled)

probability = model.predict_proba(input_scaled)


# ==============================
# 9. Display result
# ==============================

print("\n================================")
print("           RESULT")
print("================================")

if prediction[0] == 1:
    print("Prediction: Stroke")
else:
    print("Prediction: No Stroke")

print(
    f"Model probability for stroke: "
    f"{probability[0][1] * 100:.2f}%"
)

print("================================")

print("\nNote:")
print("This is an educational machine-learning project.")
print("It should NOT be used as a medical diagnosis.")