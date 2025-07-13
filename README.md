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

<img width="839" height="1061" alt="螢幕截圖 2025-07-13 下午2 39 40" src="https://github.com/user-attachments/assets/7c0e227c-c48e-4c25-9974-8f7d17d7409e" />

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

```json
{  
  "age": 35,  
  "bmi": 24.5,  
  "sex": 1,  
  "smoker": 0,  
  "steps": 12050,  
  "sleepHours": 7.5,  
  "heartRate": 65  
}
```

* **age** (number, required): User's age.  
* **bmi** (number, required): User's Body Mass Index.  
* **sex** (integer): User's gender (1 for male, 0 for female). Defaults to 1\.  
* **smoker** (integer): Smoking status (1 for smoker, 0 for non-smoker). Defaults to 0\.  
* **steps** (integer): Average daily steps. Defaults to 5000\.  
* **sleepHours** (number): Average hours of sleep per night. Defaults to 7\.  
* **heartRate** (number): Average resting heart rate. Defaults to 72\.

#### **Success Response (200 OK)**

**Example (Approved with Surcharge):**

```json
{  
  "age_group": 4,  
  "base_premium": 26571.271484375,  
  "bmi_category": "normal",  
  "discount_rate": "-50.0%",  
  "exercise_frequency": 0,  
  "final_premium": 39856.91,  
  "health_assessment": "Approved! Moderate risk applicant - 150% premium applied",  
  "health_model_used": "Neural Network",  
  "health_score": 62.1,  
  "raw_model_output": 62.1021,  
  "recommendation_list": {  
    "error": null,  
    "plans": [  
      {  
        "certification-no": "S00012-01-000-03",  
        "company-name": "Bolttech Insurance (Hong Kong) Company Limited",  
        "plan-doc-url": "https://www.vhis.gov.hk/doc/certifiedplan/sp/S00012/S00012-01-000-03-PlanDoc-e.pdf",  
        "plan-name": "VChoice Voluntary Health Insurance Plan",  
        "premium": 6736  
      },  
      {  
        "certification-no": "S00014-01-000-02",  
        "company-name": "AXA China Region Insurance Company (Bermuda) Limited",  
        "plan-doc-url": "https://www.vhis.gov.hk/doc/certifiedplan/sp/S00014/S00014-01-000-02-PlanDoc-e.pdf",  
        "plan-name": "AXA WiseGuard Medical Insurance Plan",  
        "premium": 7511  
      }  
    ]  
  },  
  "steps_converted": 1000  
}
```

**Example (Rejected):**

```json
{  
  "base_premium": 32040.5,  
  "health_score": 38.7,  
  "discount_rate": "N/A",  
  "final_premium": 0.0,  
  "health_assessment": "Reject! Application declined due to critical health risk",  
  "recommendation_list": {  
    "error": "Application rejected, no plans available.",  
    "plans": []  
  },  
  "age_group": 4,  
  "bmi_category": "obese",  
  "exercise_frequency": 0,  
  "health_model_used": "Neural Network",  
  "raw_model_output": 38.7,  
  "steps_converted": 1500  
}
```

#### **Error Response (400 Bad Request)**

```json
{  
  "error": "Missing required parameters",  
  "expected": ["age", "bmi"],  
  "received": ["smoker"]  
}
```

## **Installation and Setup**

Follow these steps to set up and run the server in a local environment.

1. **Clone the Repository**

```shell
git clone https://github.com/smudgerwth/comp7705_model.git  
cd comp7705_model
```

2.   
   **Create a Virtual Environment**

```shell
python -m venv venv  
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3.   
   **Install Dependencies**

    Create a `requirements.txt` file with the following content:

```
Flask  
pandas  
numpy  
scikit-learn  
joblib  
tensorflow
```

4.   
   Then, run the installation command:

```shell
pip install -r requirements.txt
```

5.   
   **Place Model Files**

    Ensure the following model and scaler files are in the root directory of the project:

```
health_score_nn_model.keras  
random_forest_health_score_model.joblib  
insurance_nn_model_no_child.keras  
scaler.pkl  
preprocessor.pkl  
health_score_scaler.pkl  
```

6.   
   **Run the Server**

    For development:

```shell
python api_server_rf_health_score.py
```

7.   
   The server will start on `http://localhost:5050`.

    For production, it is recommended to use a WSGI server like Gunicorn:

```shell
gunicorn --workers 4 --bind 0.0.0.0:5050 api_server_rf_health_score:app
```

### **Optional: Retraining the Models**

If you wish to retrain the models with the original datasets, you can download them from the links below. Place the contents of each zip file into a corresponding folder within the project directory. You will then need to run the training scripts (not included in this server setup).

* CASData Training: [Download Link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yiklchan_connect_hku_hk/EY4Qu3dH-sZApc_rKJlHNyEBX4il7oH7tbVe7G6wYov0YA)  
* HealthScoreData: [Download Link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yiklchan_connect_hku_hk/EYlplKsNXllCrlGiu2ZSw78BzZHBIH7KCY-qIznmeQgumw)  
* insurance\_data: [Download Link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yiklchan_connect_hku_hk/ETfkZ4ogHilIrQyLoBQQUbEBkVZlpeNciEAD54mEP5mF0A?e=cw3kfp)

## **Usage Example (cURL)**

You can test the endpoint using curl or any API client.

```shell
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
```

## **Data Privacy and Security**

We take user data privacy and security very seriously.

* **Transport Encryption**: All communication between the client and server is enforced over TLS 1.2+.  
* **Data Anonymization**: All Personally Identifiable Information (PII) is removed or hashed during data analysis and model training.  
* **Access Control**: The API endpoints should be protected by a robust authentication/authorization mechanism in a production environment.  

## **License**

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
