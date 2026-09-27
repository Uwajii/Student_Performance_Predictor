from flask import Flask, render_template, request
from model import svm_model, scaler

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction_result = None

    if request.method == "POST":

        quiz = float(request.form["quiz"])
        assignment = float(request.form["assignment"])
        attendance = float(request.form["attendance"])
        engagement = int(request.form["engagement"])

        student_data = [[quiz, assignment, attendance, engagement]]

        student_data_scaled = scaler.transform(student_data)

        prediction = svm_model.predict(student_data_scaled)

        if prediction[0] == 1:
            prediction_result = "On Track"
        else:
            prediction_result = "At Risk"

        print("Prediction:", prediction_result)

    return render_template(
        "index.html",
        prediction=prediction_result
    )


if __name__ == "__main__":
    app.run(debug=True)