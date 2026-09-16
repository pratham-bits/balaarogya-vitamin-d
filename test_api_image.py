import requests

url = "http://127.0.0.1:8000/api/v1/vitamin-d/assess-with-image"

payload = {
    "child": {
        "age_months": 24,
        "sex": "male",
        "state": "Jharkhand",
        "district": "Dhanbad",
        "residence_type": "urban",
        "season": "summer"
    }
}

with open("data/raw/test_child.jpg", "rb") as image:
    response = requests.post(
        url,
        data={"request": __import__("json").dumps(payload)},
        files={"image": image}
    )

print("STATUS:", response.status_code)
print(response.text)
