import os
import pandas as pd
import mysql.connector

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password=os.environ.get("MYSQL_PASSWORD"),
    database="StudentPerformanceDB"
)


# Load formative assessment data
query = """
SELECT
    Quiz_Average,
    Assignment_Average,
    Attendance,
    Engagement,
    Performance
FROM FormativeAssessment
"""

data = pd.read_sql(query, db)


# Prepare features and target
X = data[
    [
        "Quiz_Average",
        "Assignment_Average",
        "Attendance",
        "Engagement"
    ]
]

y = data["Performance"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create and train SVM model
svm_model = SVC()

svm_model.fit(
    X_train_scaled,
    y_train
)


# Test the model
y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("SVM model trained successfully.")
print("Training students:", len(X_train))
print("Testing students:", len(X_test))
print("Predictions:", y_pred)
print("Actual:", y_test.values)
print("SVM Accuracy:", accuracy)