#KartikBhaba_1178819
import sys
import joblib
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)

# Load Model
model = joblib.load("heart_disease_model.pkl")

class PredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heart Disease Risk Predictor")
        self.setGeometry(100, 100, 600, 700)  # wind size
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # input labels
        self.inputs = {}
        fields = {
            "age": "Age (years)",
            "sex": "Sex (0 = Female, 1 = Male)",
            "cp_1": "Chest Pain Type 1 (0 or 1)",
            "cp_2": "Chest Pain Type 2 (0 or 1)",
            "trestbps": "Resting Blood Pressure (mm Hg)",
            "chol": "Cholesterol (mg/dl)",
            "fbs": "Fasting Blood Sugar > 120 mg/dl (0 = False, 1 = True)",
            "restecg_1": "Rest ECG Type 1 (0 or 1)",
            "thalach": "Max Heart Rate Achieved",
            "exang": "Exercise-Induced Angina (0 = No, 1 = Yes)",
            "oldpeak": "ST Depression Induced by Exercise",
            "ca": "Number of Major Vessels (0–3)",
            "thal_1": "Thalassemia Type 1 (0 or 1)",
             "thal_2": "Thalassemia Type 2 (0 or 1)",  
        }

        for field, description in fields.items():
            row = QHBoxLayout()
            label = QLabel(description)
            input_box = QLineEdit()
            self.inputs[field] = input_box
            row.addWidget(label)
            row.addWidget(input_box)
            layout.addLayout(row)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict_heart_disease)
        layout.addWidget(self.predict_button)

       
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def predict_heart_disease(self):
        try:
            # Collect input data and handle missing inputs
            input_data = pd.DataFrame([{
                field: float(box.text()) if box.text().strip() else 0
                for field, box in self.inputs.items()
            }])

           
            if 'thal_1' not in self.inputs or not self.inputs['thal_1'].text().strip():
                input_data['thal_1'] = 0  
            
            if input_data.shape[1] != 14:
                input_data['thal_1'] = 0  

            X = input_data.values  # Convert to array format for the model

            # Predict
            prediction = model.predict(X)

            # Display result
            if prediction[0] == 1:
                result_message = "The patient is at high risk of heart disease."
            else:
                result_message = "The patient is at low risk of heart disease."

            QMessageBox.information(self, "Prediction Result", result_message)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec_())
