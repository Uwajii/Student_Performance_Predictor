
import sqlite3
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Connect to SQLite
db = sqlite3.connect("student_performance.db")


# Create table
db.execute("""
CREATE TABLE IF NOT EXISTS FormativeAssessment (
    Assessment_ID INTEGER PRIMARY KEY,
    Student_ID INTEGER,
    Quiz_Average REAL,
    Assignment_Average REAL,
    Attendance REAL,
    Engagement INTEGER,
    Performance INTEGER
)
""")


# Insert practice data only if the table is empty
count = db.execute(
    "SELECT COUNT(*) FROM FormativeAssessment"
).fetchone()[0]


if count == 0:

    students = [
        (1, 1, 45, 50, 65, 2, 0),
        (2, 2, 80, 75, 90, 4, 1),
        (3, 3, 60, 65, 75, 3, 1),
        (4, 4, 90, 85, 95, 5, 1),
        (5, 5, 55, 58, 70, 2, 0),
        (6, 6, 72, 70, 85, 4, 1),
        (7, 7, 88, 90, 96, 5, 1),
        (8, 8, 40, 45, 60, 1, 0),
        (9, 9, 67, 62, 78, 3, 1),
        (10, 10, 76, 78, 88, 4, 1),
        (11, 11, 52, 55, 68, 2, 0),
        (12, 12, 95, 92, 98, 5, 1),
        (13, 13, 61, 60, 72, 3, 1),
        (14, 14, 83, 80, 91, 4, 1),
        (15, 15, 48, 50, 62, 1, 0),
        (16, 16, 70, 68, 82, 3, 1),
        (17, 17, 92, 88, 94, 5, 1),
        (18, 18, 58, 57, 69, 2, 0),
        (19, 19, 74, 73, 86, 4, 1),
        (20, 20, 63, 66, 76, 3, 1)
    ]

    db.executemany("""
    INSERT INTO FormativeAssessment
    (
        Assessment_ID,
        Student_ID,
        Quiz_Average,
        Assignment_Average,
        Attendance,
        Engagement,
        Performance
    )
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, students)

    db.commit()


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

data = pd.read_sql_query(query, db)


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