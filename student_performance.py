import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="pass1122",
    database="StudentPerformanceDB"
)

print("Connected to MySQL successfully!")
engine = create_engine(
    "mysql+pymysql://root:pass1122@localhost/StudentPerformanceDB"
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)


# ==============================
# 1. Read Dataset from MySQL
# ==============================

query = """
SELECT
    Quiz_Average,
    Assignment_Average,
    Attendance,
    Engagement,
    Performance
FROM FormativeAssessment
"""

students = pd.read_sql(query, engine)

print(students)


# ==============================
# 2. Separate Features and Target
# ==============================

X = students[
    ["Quiz_Average", "Assignment_Average", "Attendance", "Engagement"]
]

y = students["Performance"]


# ==============================
# 3. Split Training and Testing Data
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training students:", len(X_train))
print("Testing students:", len(X_test))


# ==============================
# 4. Feature Scaling
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed.")


# ==============================
# 5. Decision Tree Model
# ==============================

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

rules = export_text(
    model,
    feature_names=list(X.columns)
)

print(rules)

prediction = model.predict(X_test)

print("Prediction:", prediction)
print("Actual:", y_test.values)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:")
print(cm)


# ==============================
# 6. KNN Model
# ==============================

knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(X_train_scaled, y_train)

knn_prediction = knn_model.predict(X_test_scaled)

print("KNN Prediction:", knn_prediction)
print("KNN Actual:", y_test.values)

knn_accuracy = accuracy_score(y_test, knn_prediction)
knn_precision = precision_score(y_test, knn_prediction)
knn_recall = recall_score(y_test, knn_prediction)

print("KNN Accuracy:", knn_accuracy)
print("KNN Precision:", knn_precision)
print("KNN Recall:", knn_recall)


# ==============================
# 7. SVM Model
# ==============================

svm_model = SVC()

svm_model.fit(X_train_scaled, y_train)

svm_prediction = svm_model.predict(X_test_scaled)

print("SVM Prediction:", svm_prediction)
print("SVM Actual:", y_test.values)

svm_accuracy = accuracy_score(y_test, svm_prediction)
svm_precision = precision_score(y_test, svm_prediction)
svm_recall = recall_score(y_test, svm_prediction)

print("SVM Accuracy:", svm_accuracy)
print("SVM Precision:", svm_precision)
print("SVM Recall:", svm_recall)
# ==============================
# 8. Predict New Student
# ==============================

quiz = float(input("Enter Quiz Average: "))
assignment = float(input("Enter Assignment Average: "))
attendance = float(input("Enter Attendance: "))
engagement = int(input("Enter Engagement (1-5): "))

new_student = pd.DataFrame([{
    "Quiz_Average": quiz,
    "Assignment_Average": assignment,
    "Attendance": attendance,
    "Engagement": engagement
}])

new_student_scaled = scaler.transform(new_student)

prediction = svm_model.predict(new_student_scaled)

if prediction[0] == 1:
    print("New Student Prediction: On Track")
else:
    print("New Student Prediction: At Risk")
    
prediction_text = "On Track" if prediction[0] == 1 else "At Risk"

insert_query = """
INSERT INTO Predictions
(Quiz_Average, Assignment_Average, Attendance, Engagement, Prediction)
VALUES (%s, %s, %s, %s, %s)
"""

connection.cursor().execute(
    insert_query,
    (quiz, assignment, attendance, engagement, prediction_text)
)

connection.commit()

print("Prediction saved to MySQL successfully!")