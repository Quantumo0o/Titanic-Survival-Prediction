# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
df = pd.read_csv("tested.csv")  # Change file path if needed
print(df.head())  # Display the first few rows

# Check for missing values
print(df.isnull().sum())

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)  # Fill missing Age with median
df['Fare'].fillna(df['Fare'].median(), inplace=True)  # Fill missing Fare with median
df.drop(columns=['Cabin'], inplace=True)  # Drop Cabin column (too many missing values)
df.dropna(subset=['Embarked'], inplace=True)  # Drop rows where 'Embarked' is missing

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Convert 'Sex' to numbers
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)  # One-hot encode 'Embarked'

# Drop unnecessary columns
df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# Feature selection & scaling
X = df.drop(columns=['Survived'])  # Features
y = df['Survived']  # Target variable

# Normalize Age & Fare
scaler = StandardScaler()
X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Plot the Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
