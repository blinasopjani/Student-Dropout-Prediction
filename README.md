# **Student Dropout Prediction**

## **Overview**
This is a **Machine Learning-based API** that predicts the risk of student dropout in online learning platforms. It uses a **Random Forest model** trained on simulated student behavioral and demographic data. The API supports both **binary classification** (active/at-risk vs dropped) and **multiclass classification** (active, at-risk, dropped).

The API is built using **Flask** and automates **preprocessing**, so new student data can be fed directly for real-time predictions.

---

## **Features**
- Binary and multiclass dropout prediction
- Automatic preprocessing of new student data
- Real-time prediction via REST API
- Easy-to-extend model and feature set
- Provides **feature importance** insights for understanding dropout drivers

---

## **Dataset**
- Simulated dataset of 5,000 students
- Features include:
  - Demographics: `age`, `region`
  - Engagement: `login_frequency`, `completed_assignments`, `completion_rate`
  - Activity: `last_activity_days_ago`, `courses_enrolled`
  - Temporal: `enroll_date`, `exam_season`
- Targets:
  - `label` → Binary: 0 = active/at-risk, 1 = dropped
  - `label_multiclass` → 0 = active, 1 = at-risk, 2 = dropped

---

## **Setup Instructions**

### Clone the repository

```bash
git clone https://github.com/blinasopjani/Student-Dropout-Prediction-.git
cd Student-Dropout-Prediction-

```
###  Run the API
```bash
python app.py

```

