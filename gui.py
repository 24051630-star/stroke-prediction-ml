import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib


# ==========================================
# LOAD MODEL FILES
# ==========================================

model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_stroke():

    try:
        gender = gender_var.get()
        age = float(age_entry.get())
        hypertension = int(hypertension_var.get())
        heart_disease = int(heart_var.get())
        ever_married = married_var.get()
        work_type = work_var.get()
        residence = residence_var.get()
        glucose = float(glucose_entry.get())
        bmi = float(bmi_entry.get())
        smoking = smoking_var.get()

        # Validation
        if gender == "Select":
            messagebox.showerror("Input Error", "Please select gender.")
            return

        if ever_married == "Select":
            messagebox.showerror("Input Error", "Please select married status.")
            return

        if work_type == "Select":
            messagebox.showerror("Input Error", "Please select work type.")
            return

        if residence == "Select":
            messagebox.showerror("Input Error", "Please select residence type.")
            return

        if smoking == "Select":
            messagebox.showerror("Input Error", "Please select smoking status.")
            return

        if age <= 0:
            messagebox.showerror("Input Error", "Age must be greater than 0.")
            return

        if glucose <= 0:
            messagebox.showerror(
                "Input Error",
                "Glucose level must be greater than 0."
            )
            return

        if bmi <= 0:
            messagebox.showerror(
                "Input Error",
                "BMI must be greater than 0."
            )
            return

        # Create input data
        input_data = pd.DataFrame({
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "ever_married": [ever_married],
            "work_type": [work_type],
            "Residence_type": [residence],
            "avg_glucose_level": [glucose],
            "bmi": [bmi],
            "smoking_status": [smoking]
        })

        # Encode
        input_data = pd.get_dummies(
            input_data,
            drop_first=True
        )

        # Match model features
        input_data = input_data.reindex(
            columns=features,
            fill_value=0
        )

        input_data = input_data.astype(float)

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        # Show result
        if prediction == 1:
            result_label.config(
                text="STROKE PREDICTED"
            )
        else:
            result_label.config(
                text="NO STROKE PREDICTED"
            )

        probability_label.config(
            text=f"Model Score: {probability * 100:.2f}%"
        )

    except ValueError:
        messagebox.showerror(
            "Input Error",
            "Please enter valid numbers for Age, Glucose and BMI."
        )

    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Something went wrong:\n{e}"
        )


# ==========================================
# CLEAR FUNCTION
# ==========================================

def clear_fields():

    age_entry.delete(0, tk.END)
    glucose_entry.delete(0, tk.END)
    bmi_entry.delete(0, tk.END)

    gender_var.set("Select")
    hypertension_var.set("0")
    heart_var.set("0")
    married_var.set("Select")
    work_var.set("Select")
    residence_var.set("Select")
    smoking_var.set("Select")

    result_label.config(text="")
    probability_label.config(text="")


# ==========================================
# MAIN WINDOW
# ==========================================

root = tk.Tk()

root.title("Stroke Prediction System")

# Smaller window
root.geometry("620x680")

root.resizable(False, False)


# ==========================================
# TITLE
# ==========================================

title = tk.Label(
    root,
    text="Stroke Prediction System",
    font=("Arial", 22, "bold")
)

title.pack(pady=(10, 2))


subtitle = tk.Label(
    root,
    text="Machine Learning Based Classification",
    font=("Arial", 10)
)

subtitle.pack(pady=(0, 8))


# ==========================================
# INPUT FRAME
# ==========================================

input_frame = tk.LabelFrame(
    root,
    text=" Patient Information ",
    font=("Arial", 11, "bold"),
    padx=15,
    pady=8
)

input_frame.pack(
    padx=30,
    pady=3,
    fill="x"
)


# ==========================================
# GENDER
# ==========================================

tk.Label(
    input_frame,
    text="Gender:",
    font=("Arial", 10)
).grid(
    row=0,
    column=0,
    sticky="w",
    pady=3
)

gender_var = tk.StringVar(value="Select")

gender_box = ttk.Combobox(
    input_frame,
    textvariable=gender_var,
    values=["Select", "Male", "Female", "Other"],
    state="readonly",
    width=27
)

gender_box.grid(
    row=0,
    column=1,
    padx=15,
    pady=3
)


# ==========================================
# AGE
# ==========================================

tk.Label(
    input_frame,
    text="Age:",
    font=("Arial", 10)
).grid(
    row=1,
    column=0,
    sticky="w",
    pady=3
)

age_entry = tk.Entry(
    input_frame,
    width=30
)

age_entry.grid(
    row=1,
    column=1,
    padx=15,
    pady=3
)


# ==========================================
# HYPERTENSION
# ==========================================

tk.Label(
    input_frame,
    text="Hypertension:",
    font=("Arial", 10)
).grid(
    row=2,
    column=0,
    sticky="w",
    pady=3
)

hypertension_var = tk.StringVar(value="0")

hypertension_box = ttk.Combobox(
    input_frame,
    textvariable=hypertension_var,
    values=["0", "1"],
    state="readonly",
    width=27
)

hypertension_box.grid(
    row=2,
    column=1,
    padx=15,
    pady=3
)


# ==========================================
# HEART DISEASE
# ==========================================

tk.Label(
    input_frame,
    text="Heart Disease:",
    font=("Arial", 10)
).grid(
    row=3,
    column=0,
    sticky="w",
    pady=3
)

heart_var = tk.StringVar(value="0")

heart_box = ttk.Combobox(
    input_frame,
    textvariable=heart_var,
    values=["0", "1"],
    state="readonly",
    width=27
)

heart_box.grid(
    row=3,
    column=1,
    padx=15,
    pady=3
)


# ==========================================
# EVER MARRIED
# ==========================================

tk.Label(
    input_frame,
    text="Ever Married:",
    font=("Arial", 10)
).grid(
    row=4,
    column=0,
    sticky="w",
    pady=3
)

married_var = tk.StringVar(value="Select")

married_box = ttk.Combobox(
    input_frame,
    textvariable=married_var,
    values=["Select", "Yes", "No"],
    state="readonly",
    width=27
)

married_box.grid(
    row=4,
    column=1,
    padx=15,
    pady=3
)


# ==========================================
# WORK TYPE
# ==========================================

tk.Label(
    input_frame,
    text="Work Type:",
    font=("Arial", 10)
).grid(
    row=5,
    column=0,
    sticky="w",
    pady=3
)

work_var = tk.StringVar(value="Select")

work_box = ttk.Combobox(
    input_frame,
    textvariable=work_var,
    values=[
        "Select",
        "Private",
        "Self-employed",
        "Government",
        "children",
        "Never_worked"
    ],
    state="readonly",
    width=27
)

work_box.grid(
    row=5,
    column=1,
    padx=15,
    pady=3
)


# ==========================================
# RESIDENCE
# ==========================================

tk.Label(
    input_frame,
    text="Residence Type:",
    font=("Arial", 10)
).grid(
    row=6,
    column=0,
    sticky="w",
    pady=3
)

residence_var = tk.StringVar(value="Select")

residence_box = ttk.Combobox(
    input_frame,
    textvariable=residence_var,
    values=["Select", "Urban", "Rural"],
    state="readonly",
    width=27
)

residence_box.grid(
    row=6,
    column=1,
    padx=15,
    pady=3
)


# ==========================================
# GLUCOSE
# ==========================================

tk.Label(
    input_frame,
    text="Average Glucose:",
    font=("Arial", 10)
).grid(
    row=7,
    column=0,
    sticky="w",
    pady=3
)

glucose_entry = tk.Entry(
    input_frame,
    width=30
)

glucose_entry.grid(
    row=7,
    column=1,
    padx=15,
    pady=3
)


# ==========================================
# BMI
# ==========================================

tk.Label(
    input_frame,
    text="BMI:",
    font=("Arial", 10)
).grid(
    row=8,
    column=0,
    sticky="w",
    pady=3
)

bmi_entry = tk.Entry(
    input_frame,
    width=30
)

bmi_entry.grid(
    row=8,
    column=1,
    padx=15,
    pady=3
)


# ==========================================
# SMOKING STATUS
# ==========================================

tk.Label(
    input_frame,
    text="Smoking Status:",
    font=("Arial", 10)
).grid(
    row=9,
    column=0,
    sticky="w",
    pady=3
)

smoking_var = tk.StringVar(value="Select")

smoking_box = ttk.Combobox(
    input_frame,
    textvariable=smoking_var,
    values=[
        "Select",
        "formerly smoked",
        "never smoked",
        "smokes",
        "Unknown"
    ],
    state="readonly",
    width=27
)

smoking_box.grid(
    row=9,
    column=1,
    padx=15,
    pady=3
)


# ==========================================
# BUTTONS
# ==========================================

button_frame = tk.Frame(root)

button_frame.pack(pady=8)


predict_button = tk.Button(
    button_frame,
    text="Predict Stroke",
    command=predict_stroke,
    font=("Arial", 11, "bold"),
    width=18,
    height=1
)

predict_button.grid(
    row=0,
    column=0,
    padx=8
)


clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_fields,
    font=("Arial", 11),
    width=10,
    height=1
)

clear_button.grid(
    row=0,
    column=1,
    padx=8
)


# ==========================================
# RESULT
# ==========================================

result_frame = tk.LabelFrame(
    root,
    text=" Prediction Result ",
    font=("Arial", 10, "bold"),
    padx=10,
    pady=5
)

result_frame.pack(
    padx=30,
    pady=2,
    fill="x"
)


result_label = tk.Label(
    result_frame,
    text="",
    font=("Arial", 17, "bold")
)

result_label.pack(pady=2)


probability_label = tk.Label(
    result_frame,
    text="",
    font=("Arial", 10)
)

probability_label.pack(pady=2)


# ==========================================
# DISCLAIMER
# ==========================================

disclaimer = tk.Label(
    root,
    text="Educational project only. Not a medical diagnosis.",
    font=("Arial", 8)
)

disclaimer.pack(pady=6)


# ==========================================
# START GUI
# ==========================================

root.mainloop()