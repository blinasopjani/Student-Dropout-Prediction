import requests
import pandas as pd

# Flask API URL
url = "http://127.0.0.1:5000/predict"

# Example data: multiple students
students = [
    {
        "student_id": "STU5001",
        "age": 22,
        "region": "Doha",
        "enroll_date": "2024-05-10",
        "exam_season": 0,
        "courses_enrolled": 3,
        "completed_assignments": 5,
        "completion_rate": 0.3,
        "login_frequency": 2,
        "last_activity_days_ago": 7
    },
    {
        "student_id": "STU5002",
        "age": 25,
        "region": "Baghdad",
        "enroll_date": "2024-08-15",
        "exam_season": 1,
        "courses_enrolled": 5,
        "completed_assignments": 10,
        "completion_rate": 0.6,
        "login_frequency": 4,
        "last_activity_days_ago": 2
    },
    {
        "student_id": "STU5003",
        "age": 20,
        "region": "Other",
        "enroll_date": "2024-03-20",
        "exam_season": 0,
        "courses_enrolled": 2,
        "completed_assignments": 2,
        "completion_rate": 0.1,
        "login_frequency": 0.5,
        "last_activity_days_ago": 14
    }
]

# Send POST request
response = requests.post(url, json=students)

# Parse JSON response
predictions = response.json()
print("Raw API response:")
print(predictions)

# Optional: create a neat DataFrame for visualization
df_results = pd.DataFrame({
    "student_id": [s["student_id"] for s in students],
    "binary_prediction": predictions["binary_prediction"],
    "multiclass_prediction": predictions["multiclass_prediction"]
})

print("\nPredictions Table:")
print(df_results)