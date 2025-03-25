# **Titanic Survival Prediction Model 🚢**

This project builds a **machine learning model** to predict whether a passenger survived the Titanic disaster using the famous Titanic dataset. The model processes the dataset, cleans it, and applies a **Random Forest Classifier** to predict survival outcomes.

---

## **📌 Features**
- **Data Cleaning & Preprocessing** (Handling missing values, encoding categorical variables)
- **Feature Scaling** (Normalizing `Age` and `Fare`)
- **Machine Learning Model** (Random Forest Classifier)
- **Performance Evaluation** (Accuracy, Precision, Recall, F1-score)
- **Confusion Matrix Visualization** (Heatmap to analyze model errors)

---

## **📂 Dataset Description**
The dataset consists of the following features:

| Column Name   | Description |
|--------------|-------------|
| **PassengerId** | Unique ID of the passenger |
| **Survived** | Survival status (0 = No, 1 = Yes) |
| **Pclass** | Ticket class (1st, 2nd, 3rd) |
| **Name** | Passenger name |
| **Sex** | Gender (male/female) |
| **Age** | Passenger’s age |
| **SibSp** | Number of siblings/spouses aboard |
| **Parch** | Number of parents/children aboard |
| **Ticket** | Ticket number |
| **Fare** | Ticket fare |
| **Cabin** | Cabin number (many missing values) |
| **Embarked** | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## **🛠 Installation**
Make sure you have **Python 3.x** installed, then install the required libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

## **🚀 How to Run**
1. **Clone this repository** (or download the script).
2. **Place the `titanic.csv` dataset** in the same folder.
3. **Run the script** using:
   ```bash
   python titanic_survival.py
   ```
4. **Check the output metrics and confusion matrix visualization.**

---

## **🔍 Model Workflow**
1. **Load Data** → Read the dataset.
2. **Data Cleaning** → Handle missing values (`Age`, `Fare`, `Cabin`, `Embarked`).
3. **Feature Engineering** → Encode categorical variables (`Sex`, `Embarked`).
4. **Scaling** → Normalize numerical features (`Age`, `Fare`).
5. **Train-Test Split** → Split dataset into training (80%) and testing (20%).
6. **Train Model** → Use **Random Forest Classifier** to learn survival patterns.
7. **Evaluate Model** → Calculate accuracy, precision, recall, and F1-score.
8. **Visualize Results** → Plot a confusion matrix heatmap.

---

## **📊 Expected Output**
```
Accuracy: 0.82
Precision: 0.79
Recall: 0.75
F1 Score: 0.77
```
🎯 **Confusion Matrix:** A heatmap is displayed to show correct and incorrect predictions.

---

## **📈 Possible Improvements**
✔ **Hyperparameter tuning** (optimize Random Forest settings).  
✔ **Feature engineering** (create new meaningful features).  
✔ **Try other models** (Logistic Regression, XGBoost, Neural Networks).  

---

## **📜 License**
This project is open-source and free to use.

---

## **💬 Need Help?**
Feel free to ask if you need any modifications or explanations! 🚀 Happy coding! 😊
