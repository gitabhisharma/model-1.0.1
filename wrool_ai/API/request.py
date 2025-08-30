import requests
import json

# Text analysis
response = requests.post(
    "http://localhost:8000/analyze/text",
    json={"text": "This is amazing!"}
)
print(response.json())

# Image prediction
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict/image",
        files={"file": f}
    )
print(response.json())