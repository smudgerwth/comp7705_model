# **AI Health Insurance Prediction \- Model and Server**

An AI-powered backend server that utilizes user health data from mobile & wearable devices to calculate a health score, predict insurance premiums, and provide a list of recommended insurance plans.

## **Project Overview**

This project aims to deliver an innovative Insurtech solution. By analyzing a user's daily health metrics, our AI models can:

1. **Assess a Personalized Health Score**: Quantify the user's current health status.  
2. **Dynamically Predict Insurance Premiums**: Offer premium discounts based on the health score, incentivizing users to maintain a healthy lifestyle.  
3. **Recommend Suitable Insurance Plans**: Provide customized insurance product suggestions based on the user's profile and health analysis.

The system is designed to provide insurance companies with a more accurate risk assessment tool while rewarding end-users for their healthy habits.

## **System Architecture**

The data flow is designed for efficient and secure processing of user data:

\+-----------------+      \+-----------------+      \+----------------------+      \+------------------+  
| User's Device   |      | Your App        |      | Backend Server (API) |      | AI Models        |  
| (Phone/Wearable)|-----\>| (Data Collection) |-----\>| (Data Processing)    |-----\>| (TF/Scikit-learn)|  
\+-----------------+      \+-----------------+      \+----------------------+      \+------------------+  
      ^                                                  |                             |  
      | (Results)                                        | (Storage/Cache)             | (Calculate Score/Premium)  
      |                                                  v                             v  
      \+--------------------------------------------------+                             \+------------------+  
                                                       |                             |  
                                                       \+-----------------------------+

1. **Data Collection**: The client application securely gathers health data from the user's phone and wearable devices (e.g., Apple Watch, Fitbit).  
2. **API Request**: The app sends the data via an encrypted HTTPS request to the backend server.  
3. **Model Prediction**: The server preprocesses the data and feeds it into the AI models for analysis.  
4. **Return Results**: The models return the health score and base premium. The server then calculates the final premium based on a discount logic and returns the complete analysis to the app.

## **AI Model Details**

Our core prediction engine consists of two main models, with a configurable option for the health score calculation.

* **1\. Health Score Model (Configurable)**: Predicts a user's health score.  
  * **Option A (Neural Network)**: A model built with TensorFlow/Keras (health\_score\_nn\_model.keras).  
  * **Option B (Random Forest)**: A model built with Scikit-learn (random\_forest\_health\_score\_model.joblib).  
  * **Input Features**:  
    * Age: User's age.  
    * BMI: User's Body Mass Index.  
    * Exercise\_Frequency: Derived from steps (0-6 scale).  
    * Sleep\_Hours: Average hours of sleep.  
    * Smoking\_Status: 1 for smoker, 0 for non-smoker.  
* **2\. Insurance Premium Model**: Predicts the base insurance premium before any health-based discounts.  
  * **Model Type**: A Neural Network model (insurance\_nn\_model\_no\_child.keras).  
  * **Input Features**:  
    * age: User's age.  
    * sex: 1 for male, 0 for female.  
    * bmi: User's Body Mass Index.  
    * smoker: 1 for smoker, 0 for non-smoker.  
    * *Engineered Features*: age\_bmi, bmi\_smoker, age\_smoker, age\_group, bmi\_category.

## **API Documentation**

### **POST /predict**

This is the core endpoint that receives user health data and returns a full prediction analysis.

#### **Request Body**

The request must be a JSON object with the following structure. Fields marked with \* are required.

{  
  "age": 35,  
  "bmi": 24.5,  
  "sex": 1,  
  "smoker": 0,  
  "steps": 12050,  
  "sleepHours": 7.5,  
  "heartRate": 65  
}

* age\* (number): User's age.  
* bmi\* (number): User's Body Mass Index.  
* sex (integer): User's gender (1 for male, 0 for female). Defaults to 1\.  
* smoker (integer): Smoking status (1 for smoker, 0 for non-smoker). Defaults to 0\.  
* steps (integer): Average daily steps. Defaults to 5000\.  
* sleepHours (number): Average hours of sleep per night. Defaults to 7\.  
* heartRate (number): Average resting heart rate. Defaults to 72\.

#### **Success Response (200 OK)**

The response is a JSON object containing the full analysis.

**Example (Approved with Discount):**

{  
    "base\_premium": 1250.75,  
    "health\_score": 88.5,  
    "discount\_rate": "20.0%",  
    "final\_premium": 1000.60,  
    "health\_assessment": "Approved\! Preferred risk applicant \- 20% discount",  
    "recommendation\_list": \[  
        {  
            "plan\_name": "Comprehensive Health Guard",  
            "premium": "1000",  
            "coverage\_details": "Covers hospitalization, surgery, and critical illness."  
        }  
    \],  
    "age\_group": 2,  
    "bmi\_category": "normal",  
    "exercise\_frequency": 6,  
    "health\_model\_used": "Neural Network",  
    "raw\_model\_output": 88.5,  
    "steps\_converted": 12050  
}

**Example (Rejected):**

{  
    "base\_premium": 1800.50,  
    "health\_score": 35.2,  
    "discount\_rate": "N/A",  
    "final\_premium": 0.0,  
    "health\_assessment": "Reject\! Application declined due to critical health risk",  
    "recommendation\_list": \[\],  
    "age\_group": 3,  
    "bmi\_category": "overweight",  
    "exercise\_frequency": 1,  
    "health\_model\_used": "Neural Network",  
    "raw\_model\_output": 35.2,  
    "steps\_converted": 3500  
}

#### **Error Response (400 Bad Request)**

If required data is missing or invalid.

{  
  "error": "Missing required parameters",  
  "expected": \["age", "bmi"\],  
  "received": \["smoker"\]  
}

## **Installation and Setup**

Follow these steps to set up and run the server in a local environment.

1. **Clone the Repository**  
   git clone https://github.com/your-username/health-insurance-ai.git  
   cd health-insurance-ai

2. **Create a Virtual Environment**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

3. Install Dependencies  
   Create a requirements.txt file with the following content:  
   Flask  
   pandas  
   numpy  
   scikit-learn  
   joblib  
   tensorflow

   Then, run the installation command:  
   pip install \-r requirements.txt

4. Place Model Files  
   Ensure the following model and scaler files are in the root directory of the project:  
   * health\_score\_nn\_model.keras  
   * random\_forest\_health\_score\_model.joblib  
   * insurance\_nn\_model\_no\_child.keras  
   * scaler.pkl  
   * preprocessor.pkl  
   * health\_score\_scaler.pkl  
5. Run the Server  
   For development:  
   python api\_server\_rf\_health\_score.py

   The server will start on http://localhost:5050.  
   For production, it is recommended to use a WSGI server like Gunicorn:  
   gunicorn \--workers 4 \--bind 0.0.0.0:5050 api\_server\_rf\_health\_score:app

## **Usage Example (cURL)**

You can test the endpoint using curl or any API client.

curl \-X POST http://localhost:5050/predict \\  
\-H "Content-Type: application/json" \\  
\-d '{  
    "age": 40,  
    "bmi": 28,  
    "sex": 0,  
    "smoker": 0,  
    "steps": 7500,  
    "sleepHours": 6.5,  
    "heartRate": 75  
}'

## **Data Privacy and Security**

We take user data privacy and security very seriously.

* **Transport Encryption**: All communication between the client and server is enforced over TLS 1.2+.  
* **Data Anonymization**: All Personally Identifiable Information (PII) is removed or hashed during data analysis and model training.  
* **Access Control**: The API endpoints should be protected by a robust authentication/authorization mechanism in a production environment.  
* **Compliance**: Our data handling procedures are designed to comply with GDPR and other regional data protection regulations.

## **Contributing**

We welcome contributions to this project\! Please refer to the CONTRIBUTING.md file for detailed contribution guidelines, code style, and the Pull Request process.

## **License**

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).