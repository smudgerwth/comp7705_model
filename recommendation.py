import json
import random

MIN_VALUE = 0
MAX_VALUE = 50000
BASE_URL = "https://www.vhis.gov.hk"
PREMIUM_JSON = "VHIS/plan-premium.json"
STANDARD_PLANS_JSON = "VHIS/standard-plans.json"

def generate_standalone_letters(sex: int, smoking: int) -> str:
    first_letter = 'S'
    second_letter = 'M' if sex == 1 else 'F' if sex == 0 else None
    third_letter = 'Y' if smoking == 1 else 'N' if smoking == 0 else None
    if second_letter is None or third_letter is None:
        return None
    return f"{first_letter}{second_letter}{third_letter}"

def get_certification_numbers_within_range(json_data, age: int, code: str, MIN_VALUE: int, MAX_VALUE: int):
    print(f"Searching for certification numbers for age: {age}, code: {code} within range {MIN_VALUE}-{MAX_VALUE}")
    cert_nos_premium = []
    # Iterate with index for debugging
    for idx, plan in enumerate(json_data.get("certified-plans", [])):
        try:
            value = plan.get("premium", {}).get(code, {}).get(str(age))
            if (value is not None and
                isinstance(value, (int, float)) and
                MIN_VALUE <= value <= MAX_VALUE):
                cert_no = plan.get("certification-no")
                if cert_no and cert_no.startswith('S'):  # Filter standalone plans here
                    cert_nos_premium.append((cert_no, value))
        except Exception as e:
            pass
            # print(f"Error processing plan at index {idx}: {str(e)}")

    if len(cert_nos_premium) > 3:
        cert_nos_premium = random.sample(cert_nos_premium, 3)
    elif len(cert_nos_premium) == 0:
        return None
    return cert_nos_premium

def get_plan_details_by_certification(json_data, cert_no: str):
    for plan in json_data.get("certified-plans", []):
        plan_info = plan.get("plan-info-certified", [])
        if not isinstance(plan_info, list):
            print(f"Warning: plan-info-certified is not a list for certification-no {cert_no}")
            continue
        for info in plan_info:
            if info.get("certification-no") == cert_no:
                return (
                    plan.get("company-name", {}).get("en"),
                    plan.get("plan-name", {}).get("en"),
                    BASE_URL + info.get("plan-doc-url", {}).get("en")
                )
    return None

def get_plan_details(age: int, sex: int, smoking: int):
    # Initialize result dictionary
    result = {"plans": [], "error": None}
    
    # Generate standalone letters
    code = generate_standalone_letters(sex, smoking)
    if not code:
        result["error"] = "Invalid sex or smoking status"
        return result
    
    # Load plan-premium.json
    try:
        with open(PREMIUM_JSON, "r") as file:
            premium_data = json.load(file)
    except FileNotFoundError:
        result["error"] = f"Failed to load {PREMIUM_JSON}: File not found"
        return result
    except json.JSONDecodeError:
        result["error"] = f"Failed to load {PREMIUM_JSON}: Invalid JSON format"
        return result
    
    # Get certification numbers within range
    cert_nos_premium = get_certification_numbers_within_range(premium_data, age, code, MIN_VALUE, MAX_VALUE)
    
    if not cert_nos_premium:
        result["error"] = f"No certification numbers found for age {age} and code {code} within range {MIN_VALUE}-{MAX_VALUE}"
        return result
    
    # Load standard-plans.json
    try:
        with open(STANDARD_PLANS_JSON, "r") as file:
            plan_data = json.load(file)
    except FileNotFoundError:
        result["error"] = f"Failed to load {STANDARD_PLANS_JSON}: File not found"
        return result
    except json.JSONDecodeError:
        result["error"] = f"Failed to load {STANDARD_PLANS_JSON}: Invalid JSON format"
        return result
    
    # Get plan details for each certification number
    for cert_no, premium in cert_nos_premium:
        details = get_plan_details_by_certification(plan_data, cert_no)
        if details:
            plan_details = {
                "certification-no": cert_no,
                "company-name": details[0],
                "plan-name": details[1],
                "plan-doc-url": details[2],
                "premium": premium
            }
            result["plans"].append(plan_details)
        else:
            print(f"No plan found for certification number {cert_no}.")
    
    if not result["plans"]:
        result["error"] = "No plan details found for the given certification numbers"
    
    return result

if __name__ == "__main__":
    age = 30  # Example age
    sex = 1   # Example sex (1 for male, 0 for female)
    smoking = 0  # Example smoking status (1 for smoker, 0 for non-smoker)

    plan_details = get_plan_details(age, sex, smoking)
    print(json.dumps(plan_details, indent=2))