import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_new_data(new_df, scaler):
    """
    Preprocess new student data for prediction
    Input: new_df - DataFrame with raw student data
           scaler - loaded scaler for numeric features
    Output: preprocessed DataFrame ready for model prediction
    """
    # Encode region
    new_df = pd.get_dummies(new_df, columns=['region'], drop_first=True)

    # Convert enroll_date to month
    new_df['enroll_date'] = pd.to_datetime(new_df['enroll_date'])
    new_df['enroll_month'] = new_df['enroll_date'].dt.month
    new_df = new_df.drop(['enroll_date', 'student_id'], axis=1, errors='ignore')

    # Feature engineering 
    # engagement score
    if 'login_frequency' in new_df.columns and 'completion_rate' in new_df.columns:
        new_df['engagement_score'] = new_df['login_frequency'] * new_df['completion_rate']

    # recency
    if 'last_activity_days_ago' in new_df.columns:
        new_df['recency'] = new_df['last_activity_days_ago']

    # low engagement flag
    new_df['low_engagement_flag'] = np.where(
        (new_df['login_frequency'] < 2) & (new_df['completion_rate'] < 0.3), 1, 0
    )

    # course load
    if 'courses_enrolled' in new_df.columns:
        new_df['course_load'] = pd.cut(new_df['courses_enrolled'], bins=[0, 2, 5, 7], labels=['light','medium','heavy'])
        new_df = pd.get_dummies(new_df, columns=['course_load'], drop_first=True)

    # Scale numeric features 
    numeric_cols = ['age', 'login_frequency', 'completed_assignments', 
                    'completion_rate', 'last_activity_days_ago', 'courses_enrolled']
    for col in numeric_cols:
        if col not in new_df.columns:
            new_df[col] = 0  # default if missing

    new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])

    return new_df