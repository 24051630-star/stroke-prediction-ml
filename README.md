# Stroke Prediction System 🧠

## 📌 Project Overview

The **Stroke Prediction System** is a Machine Learning project that predicts whether a person belongs to the **stroke** or **non-stroke** class based on different health-related features.

The project includes data analysis, data cleaning, preprocessing, handling of imbalanced data using **SMOTE**, machine learning model training, evaluation, and a simple **Graphical User Interface (GUI)** for making predictions.

> **Note:** This project is created for educational purposes only. It is not a medical diagnosis or a substitute for professional medical advice.

---

## 🎯 Objective

The main objectives of this project are:

* Understand and analyze a real-world healthcare dataset.
* Perform Exploratory Data Analysis (EDA).
* Clean and preprocess the dataset.
* Handle missing values.
* Convert categorical data into numerical features.
* Handle class imbalance using SMOTE.
* Train and compare Machine Learning models.
* Evaluate model performance using suitable metrics.
* Save the trained model for future predictions.
* Create a simple GUI for user input and prediction.

---

## 📊 Dataset

The dataset contains **43,400 records** and **12 columns**.

### Features

| Feature             | Description                              |
| ------------------- | ---------------------------------------- |
| `id`                | Unique identifier                        |
| `gender`            | Gender of the person                     |
| `age`               | Age                                      |
| `hypertension`      | Whether the person has hypertension      |
| `heart_disease`     | Whether the person has heart disease     |
| `ever_married`      | Whether the person has ever been married |
| `work_type`         | Type of employment                       |
| `Residence_type`    | Urban or Rural residence                 |
| `avg_glucose_level` | Average glucose level                    |
| `bmi`               | Body Mass Index                          |
| `smoking_status`    | Smoking status                           |
| `stroke`            | Target variable                          |

### Target Variable

The target variable is:

```text
stroke
```

Where:

```text
0 = No Stroke
1 = Stroke
```

---

## 🔍 Exploratory Data Analysis

EDA was performed to understand the dataset before training the models.

The analysis included:

* First few rows of the dataset
* Dataset shape
* Column names
* Data types
* Statistical summary
* Missing values
* Duplicate records
* Unique values
* Target distribution
* Categorical feature distributions
* Age distribution
* BMI distribution
* Glucose level distribution
* Outlier detection
* Correlation analysis

Several visualizations were also created to understand the relationship between different features and stroke.

---

## 🧹 Data Cleaning

The dataset contained missing values in:

* `bmi`
* `smoking_status`

### BMI

Missing BMI values were replaced using the **median BMI**.

### Smoking Status

Missing smoking status values were replaced with:

```text
Unknown
```

After cleaning, the dataset contained **no missing values**.

The `id` column was removed before model training because it is only an identifier and does not provide useful predictive information.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

1. Removed the `id` column.
2. Separated features and target.
3. Converted categorical variables into numerical variables using **One-Hot Encoding**.
4. Split the dataset into training and testing sets.
5. Used stratified splitting to maintain the target distribution.
6. Applied SMOTE only to the training data.
7. Scaled the data for Logistic Regression.

### Dataset Split

```text
Training data: 80%
Testing data: 20%
```

Training dataset:

```text
34,720 records
```

Testing dataset:

```text
8,680 records
```

---

## ⚖️ Handling Class Imbalance

The dataset contains significantly more non-stroke cases than stroke cases.

Before SMOTE:

```text
Non-Stroke: 34,094
Stroke:        626
```

Because of this imbalance, the model could achieve high accuracy simply by predicting most cases as non-stroke.

To address this problem, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training data.

After SMOTE:

```text
Non-Stroke: 34,094
Stroke:     34,094
```

This created a balanced training dataset.

---

## 🤖 Machine Learning Models

Two models were trained and compared:

### 1. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions.

### 2. Logistic Regression

Logistic Regression is a classification algorithm that estimates the probability of belonging to a particular class.

---

## 📈 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* Average Precision

### Random Forest

```text
Accuracy : 98.12%
ROC-AUC  : 0.7709
```

Stroke recall:

```text
1%
```

Although Random Forest achieved high accuracy, it detected very few actual stroke cases.

---

### Logistic Regression

```text
Accuracy : 75.37%
ROC-AUC  : 0.8390
Average Precision : 0.0786
```

Stroke recall:

```text
79%
```

The Logistic Regression model detected substantially more stroke cases than Random Forest.

---

## 🏆 Final Model

**Logistic Regression** was selected as the final model.

The main reason for selecting Logistic Regression was that accuracy alone was not sufficient for this imbalanced classification problem.

The Logistic Regression model achieved:

```text
ROC-AUC: 0.8390
Stroke Recall: 79%
```

This means the model was able to identify a much larger proportion of the actual stroke cases in the test dataset compared with Random Forest.

---

## 💾 Saved Model Files

The trained model and preprocessing information are saved as:

```text
stroke_model.pkl
scaler.pkl
features.pkl
```

### `stroke_model.pkl`

Contains the trained Logistic Regression model.

### `scaler.pkl`

Contains the scaler used to transform the input features.

### `features.pkl`

Contains the feature names/order used during model training.

These files allow the prediction program and GUI to use the already-trained model without training it again.

---

## 🖥️ Graphical User Interface

A simple GUI was created using **Tkinter**.

The user can enter:

* Gender
* Age
* Hypertension
* Heart Disease
* Marital Status
* Work Type
* Residence Type
* Average Glucose Level
* BMI
* Smoking Status

The application then displays the predicted class and the model's calculated stroke-class probability.

### Example Tests

| Test                                                           | Result                          |
| -------------------------------------------------------------- | ------------------------------- |
| Young person with low-risk input                               | 8.65% stroke-class probability  |
| Older person with hypertension and high BMI                    | 51.61% stroke-class probability |
| Older person with hypertension, heart disease and high glucose | 91.35% stroke-class probability |

These values are **model outputs**, not medically validated probabilities or actual medical risk percentages.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Tkinter
* Joblib
* Jupyter/VS Code/Python IDE

---

## 📁 Project Structure

```text
projectaiml/
│
├── dataset.csv
│
├── eda.py
│   └── Exploratory Data Analysis
│
├── cleaning.py
│   └── Data cleaning and missing-value handling
│
├── preprocessing.py
│   └── Encoding, SMOTE, model training and evaluation
│
├── prediction.py
│   └── Command-line prediction system
│
├── gui.py
│   └── Graphical User Interface
│
├── stroke_model.pkl
│   └── Trained Logistic Regression model
│
├── scaler.pkl
│   └── Saved feature scaler
│
├── features.pkl
│   └── Saved feature names
│
└── README.md
    └── Project documentation
```

---

## ▶️ How to Run the Project

### 1. Install required libraries

Open Command Prompt or PowerShell inside the project folder and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

---

### 2. Run EDA

```bash
python eda.py
```

This performs exploratory analysis and generates visualizations.

---

### 3. Run Data Cleaning

```bash
python cleaning.py
```

This handles missing values and prepares the dataset.

---

### 4. Train the Models

```bash
python preprocessing.py
```

This performs:

```text
Preprocessing
      ↓
Train-Test Split
      ↓
SMOTE
      ↓
Random Forest
      ↓
Logistic Regression
      ↓
Model Evaluation
      ↓
Save Model
```

---

### 5. Test Prediction

```bash
python prediction.py
```

The program will ask for user information and generate a prediction.

---

### 6. Run GUI

```bash
python gui.py
```

This opens the Stroke Prediction System graphical interface.

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
One-Hot Encoding
   ↓
Train-Test Split
   ↓
SMOTE
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Select Best Model
   ↓
Save Model
   ↓
Prediction
   ↓
GUI
```

---

## ⚠️ Limitations

This project has several limitations:

* The dataset is imbalanced.
* The model is trained on the available dataset and may not generalize to all populations.
* The project is not clinically validated.
* Model probability should not be interpreted as an actual medical risk percentage.
* More advanced feature engineering and model tuning could improve the system.
* Real-world medical applications require professional validation and appropriate clinical data.

---

## 🚀 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Better handling of class imbalance
* Trying additional classification algorithms
* Cross-validation
* Feature engineering
* Improved GUI design
* Web-based deployment
* Cloud deployment
* Model explainability using SHAP or similar techniques
* Better probability calibration
* Larger and more diverse datasets

---

## 📸 Screenshots

Add screenshots of the following to this section:

### EDA

```text
[Add EDA screenshot here]
```

### Model Evaluation

```text
[Add classification report / confusion matrix screenshot here]
```

### GUI

```text
[Add GUI screenshot here]
```

### Prediction Result

```text
[Add prediction result screenshot here]
```

---

## 🎓 Learning Outcomes

Through this project, I learned:

* How to work with a real-world dataset.
* How to perform EDA using Pandas and visualization libraries.
* How to identify and handle missing values.
* How categorical data is converted into numerical data.
* Why class imbalance is important in classification problems.
* How SMOTE can be used to balance training data.
* How to train and compare Machine Learning models.
* How to evaluate classification models using recall, precision, F1-score and ROC-AUC.
* How to save and reuse a trained Machine Learning model.
* How to build a basic GUI using Tkinter.
* How to create an end-to-end Machine Learning project.

---

## 👩‍💻 Author

**Jyotsana Sharma**

B.Tech Computer Science and Engineering
Kalinga Institute of Industrial Technology (KIIT)

---

## 📌 Disclaimer

This project is developed for **educational and learning purposes only**.

The predictions generated by this application should **not be used for medical diagnosis, treatment, or healthcare decisions**. Always consult a qualified healthcare professional for medical advice.
