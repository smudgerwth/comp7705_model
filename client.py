import requests

url = "http://localhost:5050/predict"
payload = {
    "age": 40,
    "bmi": 28.0,
    "sex": 1,      # 1 for female, 0 for male
    "smoker": 0    # 1 for yes, 0 for no
}

print("Sending POST request to:", url)
print("Payload:", payload)

try:
    response = requests.post(url, json=payload)
    print("Status code:", response.status_code)
    print("Response headers:", response.headers)
    print("Raw response text:", response.text)
    if response.ok:
        print("Predicted premium:", response.json()['predicted_premium'])
    else:
        print("Error:", response.text)
except Exception as e:
    print("Exception occurred:", e)