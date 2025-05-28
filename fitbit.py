import requests
from requests_oauthlib import OAuth2Session
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import json
import base64

# Fitbit API credentials (replace with your own)
CLIENT_ID = "YOUR_CLIENT_ID"  # From dev.fitbit.com
CLIENT_SECRET = "YOUR_CLIENT_SECRET"  # Keep secure
REDIRECT_URI = "http://127.0.0.1:8080"  # Local testing
AUTH_URL = "https://www.fitbit.com/oauth2/authorize"
TOKEN_URL = "https://api.fitbit.com/oauth2/token"
SCOPES = ["activity", "heartrate", "sleep"]

# Global variable to store authorization code
auth_code = None

# Simple HTTP server to handle OAuth callback
class OAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code
        query = urllib.parse.urlparse(self.path).query
        query_components = urllib.parse.parse_qs(query)
        auth_code = query_components.get("code", [None])[0]
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"Authorization complete. You can close this window.")

def start_server():
    server = HTTPServer(("127.0.0.1", 8080), OAuthHandler)
    server.handle_request()  # Handle one request (callback)
    return auth_code

def authenticate():
    fitbit = OAuth2Session(CLIENT_ID, redirect_uri=REDIRECT_URI, scope=SCOPES)
    authorization_url, _ = fitbit.authorization_url(AUTH_URL)
    print(f"Please visit this URL to authorize: {authorization_url}")

    # Start local server to capture authorization code
    auth_code = start_server()
    if not auth_code:
        raise Exception("Authorization failed: No code received")

    # Exchange code for tokens
    token = fitbit.fetch_token(
        TOKEN_URL,
        code=auth_code,
        client_secret=CLIENT_SECRET,
        include_client_id=True
    )
    return token["access_token"], token["refresh_token"]

def fetch_fitbit_data(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Fetch daily steps
    steps_response = requests.get(
        "https://api.fitbit.com/1/user/-/activities/steps/date/today/1d.json",
        headers=headers
    )
    steps_data = steps_response.json()

    # Fetch heart rate
    heart_response = requests.get(
        "https://api.fitbit.com/1/user/-/activities/heart/date/today/1d.json",
        headers=headers
    )
    heart_data = heart_response.json()

    return steps_data, heart_data

def analyze_data(steps_data, heart_data):
    # Extract relevant metrics
    steps = steps_data.get("activities-steps", [{}])[0].get("value", 0)
    resting_heart_rate = heart_data.get("activities-heart", [{}])[0].get("value", {}).get("restingHeartRate", None)

    # Health advice
    advice = []
    if steps and int(steps) < 5000:
        advice.append("Your step count is low. Aim for 10,000 steps/day to improve cardiovascular health.")
    else:
        advice.append("Great job staying active! Maintain or increase your step count for optimal health.")
    
    if resting_heart_rate and resting_heart_rate > 80:
        advice.append("Your resting heart rate is high. Consider stress management or consulting a doctor.")
    elif resting_heart_rate:
        advice.append("Your resting heart rate is healthy, indicating good fitness.")

    # Insurance recommendation
    insurance_rec = []
    if steps and int(steps) > 10000 and resting_heart_rate and resting_heart_rate < 70:
        insurance_rec.append("You qualify for wellness-focused insurance plans with premium discounts (e.g., John Hancock Vitality).")
    else:
        insurance_rec.append("Consider plans with comprehensive coverage, as your activity or heart rate suggests potential health risks.")

    return advice, insurance_rec

def main():
    try:
        # Authenticate
        access_token, refresh_token = authenticate()

        # Fetch data
        steps_data, heart_data = fetch_fitbit_data(access_token)

        # Analyze and provide recommendations
        health_advice, insurance_recommendations = analyze_data(steps_data, heart_data)
        
        print("\nHealth Advice:")
        for advice in health_advice:
            print(f"- {advice}")
        
        print("\nInsurance Recommendations:")
        for rec in insurance_recommendations:
            print(f"- {rec}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()