AI Health Insurance Prediction - Model and Server
An AI-powered backend server that utilizes user health data from mobile & wearable devices to calculate a health score, predict insurance premiums, and provide a list of recommended insurance plans.
Project Overview
This project aims to deliver an innovative Insurtech solution. By analyzing a user's daily health metrics, our AI models can:
Assess a Personalized Health Score: Quantify the user's current health status.
Dynamically Predict Insurance Premiums: Offer premium discounts based on the health score, incentivizing users to maintain a healthy lifestyle.
Recommend Suitable Insurance Plans: Provide customized insurance product suggestions based on the user's profile and health analysis.
The system is designed to provide insurance companies with a more accurate risk assessment tool while rewarding end-users for their healthy habits.
System Architecture
The data flow is designed for efficient and secure processing of user data:
+-----------------+      +-----------------+      +----------------------+      +------------------+
| User's Device   |      | Your App        |      | Backend Server (API) |      | AI Models        |
| (Phone/Wearable)|----->| (Data Collection) |----->| (Data Processing)    |----->| (TF/Scikit-learn)|
+-----------------+      +-----------------+      +----------------------+      +------------------+
      ^                                                  |                             |
      | (Results)                                        | (Storage/Cache)             | (Calculate Score/Premium)
      |                                                  v                             v
      +--------------------------------------------------+                             +------------------+
                                                       |                             |
                                                       +-----------------------------+


Data Collection: The client application securely gathers health data from the user's phone and wearable devices (e.g., Apple Watch, Fitbit).
API Request: The app sends the data via an encrypted HTTPS request to the backend server.
Model Prediction: The server preprocesses the data and feeds it into the AI models for analysis.
Return Results: The models return the health score and base premium. The server then calculates the final premium based on a discount logic and returns the complete analysis to the app.
AI Model Details
Our core prediction engine consists of two main models, with a configurable option for the health score calculation.
1. Health Score Model (Configurable): Predicts a user's health score.
Option A (Neural Network): A model built with TensorFlow/Keras (health_score_nn_model.keras).
Option B (Random Forest): A model built with Scikit-learn (random_forest_health_score_model.joblib).
Input Features:
Age: User's age.
BMI: User's Body Mass Index.
Exercise_Frequency: Derived from steps (0-6 scale).
Sleep_Hours: Average hours of sleep.
Smoking_Status: 1 for smoker, 0 for non-smoker.
2. Insurance Premium Model: Predicts the base insurance premium before any health-based discounts.
Model Type: A Neural Network model (insurance_nn_model_no_child.keras).
Input Features:
age: User's age.
sex: 1 for male, 0 for female.
bmi: User's Body Mass Index.
smoker: 1 for smoker, 0 for non-smoker.
Engineered Features: age_bmi, bmi_smoker, age_smoker, age_group, bmi_category.
API Documentation
POST /predict
This is the core endpoint that receives user health data and returns a full prediction analysis.
Request Body
The request must be a JSON object with the following structure. Fields marked with * are required.
{
  "age": 35,
  "bmi": 24.5,
  "sex": 1,
  "smoker": 0,
  "steps": 12050,
  "sleepHours": 7.5,
  "heartRate": 65
}


age* (number): User's age.
bmi* (number): User's Body Mass Index.
sex (integer): User's gender (1 for male, 0 for female). Defaults to 1.
smoker (integer): Smoking status (1 for smoker, 0 for non-smoker). Defaults to 0.
steps (integer): Average daily steps. Defaults to 5000.
sleepHours (number): Average hours of sleep per night. Defaults to 7.
heartRate (number): Average resting heart rate. Defaults to 72.
Success Response (200 OK)
The response is a JSON object containing the full analysis.
Example (Approved with Discount):
{
    "base_premium": 1250.75,
    "health_score": 88.5,
    "discount_rate": "20.0%",
    "final_premium": 1000.60,
    "health_assessment": "Approved! Preferred risk applicant - 20% discount",
    "recommendation_list": [
        {
            "plan_name": "Comprehensive Health Guard",
            "premium": "1000",
            "coverage_details": "Covers hospitalization, surgery, and critical illness."
        }
    ],
    "age_group": 2,
    "bmi_category": "normal",
    "exercise_frequency": 6,
    "health_model_used": "Neural Network",
    "raw_model_output": 88.5,
    "steps_converted": 12050
}


Example (Rejected):
{
    "base_premium": 1800.50,
    "health_score": 35.2,
    "discount_rate": "N/A",
    "final_premium": 0.0,
    "health_assessment": "Reject! Application declined due to critical health risk",
    "recommendation_list": [],
    "age_group": 3,
    "bmi_category": "overweight",
    "exercise_frequency": 1,
    "health_model_used": "Neural Network",
    "raw_model_output": 35.2,
    "steps_converted": 3500
}


Error Response (400 Bad Request)
If required data is missing or invalid.
{
  "error": "Missing required parameters",
  "expected": ["age", "bmi"],
  "received": ["smoker"]
}


Installation and Setup
Follow these steps to set up and run the server in a local environment.
Clone the Repository
git clone https://github.com/your-username/health-insurance-ai.git
cd health-insurance-ai


Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install Dependencies
Create a requirements.txt file with the following content:
Flask
pandas
numpy
scikit-learn
joblib
tensorflow

Then, run the installation command:
pip install -r requirements.txt


Place Model Files
Ensure the following model and scaler files are in the root directory of the project:
health_score_nn_model.keras
random_forest_health_score_model.joblib
insurance_nn_model_no_child.keras
scaler.pkl
preprocessor.pkl
health_score_scaler.pkl
Run the Server
For development:
python api_server_rf_health_score.py

The server will start on http://localhost:5050.
For production, it is recommended to use a WSGI server like Gunicorn:
gunicorn --workers 4 --bind 0.0.0.0:5050 api_server_rf_health_score:app


Usage Example (cURL)
You can test the endpoint using curl or any API client.
curl -X POST http://localhost:5050/predict \
-H "Content-Type: application/json" \
-d '{
    "age": 40,
    "bmi": 28,
    "sex": 0,
    "smoker": 0,
    "steps": 7500,
    "sleepHours": 6.5,
    "heartRate": 75
}'


Data Privacy and Security
We take user data privacy and security very seriously.
Transport Encryption: All communication between the client and server is enforced over TLS 1.2+.
Data Anonymization: All Personally Identifiable Information (PII) is removed or hashed during data analysis and model training.
Access Control: The API endpoints should be protected by a robust authentication/authorization mechanism in a production environment.
Compliance: Our data handling procedures are designed to comply with GDPR and other regional data protection regulations.
Contributing
We welcome contributions to this project! Please refer to the CONTRIBUTING.md file for detailed contribution guidelines, code style, and the Pull Request process.
License
This project is licensed under the MIT License.
