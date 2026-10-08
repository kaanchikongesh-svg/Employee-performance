"""
Central Configuration for Employee Performance Prediction System.
Defines feature sets, performance score thresholds, categorical choices, and UI themes.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

SAMPLE_DATA_PATH = DATA_DIR / "sample_employee_data.csv"
MODEL_FILE_PATH = MODELS_DIR / "employee_performance_model.pkl"

# Performance Categories & Thresholds
# 80-100 -> High, 60-79 -> Medium, 0-59 -> Low
THRESHOLD_HIGH = 80.0
THRESHOLD_MEDIUM = 60.0

CATEGORY_HIGH = "High"
CATEGORY_MEDIUM = "Medium"
CATEGORY_LOW = "Low"

CATEGORIES = [CATEGORY_LOW, CATEGORY_MEDIUM, CATEGORY_HIGH]

# Category Color Mapping
CATEGORY_COLORS = {
    CATEGORY_HIGH: "#10B981",    # Emerald green
    CATEGORY_MEDIUM: "#F59E0B",  # Amber / orange
    CATEGORY_LOW: "#EF4444"      # Rose / Red
}

# Categorical Feature Definitions
GENDER_OPTIONS = ["Male", "Female", "Non-Binary", "Other"]

DEPARTMENT_OPTIONS = [
    "Engineering",
    "Sales",
    "Marketing",
    "Human Resources",
    "Finance",
    "Operations",
    "Customer Support"
]

JOB_ROLE_OPTIONS = {
    "Engineering": ["Software Engineer", "Senior Software Engineer", "DevOps Engineer", "QA Engineer", "Tech Lead"],
    "Sales": ["Sales Executive", "Account Executive", "Sales Manager", "Business Development Rep"],
    "Marketing": ["Marketing Specialist", "Content Strategist", "SEO Analyst", "Growth Manager"],
    "Human Resources": ["HR Specialist", "Talent Acquisition Lead", "HR Business Partner", "Compensation Analyst"],
    "Finance": ["Financial Analyst", "Accountant", "Finance Manager", "Auditor"],
    "Operations": ["Operations Manager", "Supply Chain Analyst", "Logistics Coordinator", "Process Lead"],
    "Customer Support": ["Support Specialist", "Customer Success Manager", "Technical Support Engineer"]
}

ALL_JOB_ROLES = sorted(list({role for roles in JOB_ROLE_OPTIONS.values() for role in roles}))

PROMOTION_OPTIONS = ["No", "Yes"]

# Schema Feature Groupings
IDENTIFIER_COLUMNS = ["Employee_ID", "Name"]

NUMERICAL_FEATURES = [
    "Age",
    "Years_of_Experience",
    "Monthly_Income",
    "Job_Level",
    "Job_Satisfaction",
    "Work_Life_Balance",
    "Working_Hours_per_Week",
    "Attendance_Percentage",
    "Training_Hours",
    "Number_of_Trainings",
    "Projects_Completed",
    "Overtime_Hours",
    "Previous_Performance_Score",
    "Years_Since_Last_Promotion",
    "Team_Size"
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Department",
    "Job_Role",
    "Promotion_History"
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_SCORE_COLUMN = "Performance_Score"
TARGET_CATEGORY_COLUMN = "Performance_Category"

# Feature Labels & Units for UI
FEATURE_METADATA = {
    "Age": {"min": 18, "max": 65, "default": 32, "step": 1, "unit": "years"},
    "Years_of_Experience": {"min": 0.0, "max": 40.0, "default": 6.5, "step": 0.5, "unit": "years"},
    "Monthly_Income": {"min": 1000.0, "max": 30000.0, "default": 6500.0, "step": 250.0, "unit": "$/mo"},
    "Job_Level": {"min": 1, "max": 5, "default": 3, "step": 1, "unit": "level (1-5)"},
    "Job_Satisfaction": {"min": 1, "max": 5, "default": 4, "step": 1, "unit": "rating (1-5)"},
    "Work_Life_Balance": {"min": 1, "max": 5, "default": 3, "step": 1, "unit": "rating (1-5)"},
    "Working_Hours_per_Week": {"min": 20.0, "max": 80.0, "default": 42.0, "step": 1.0, "unit": "hrs/week"},
    "Attendance_Percentage": {"min": 50.0, "max": 100.0, "default": 94.0, "step": 0.5, "unit": "%"},
    "Training_Hours": {"min": 0.0, "max": 200.0, "default": 35.0, "step": 2.0, "unit": "hrs/yr"},
    "Number_of_Trainings": {"min": 0, "max": 15, "default": 3, "step": 1, "unit": "trainings"},
    "Projects_Completed": {"min": 0, "max": 40, "default": 8, "step": 1, "unit": "projects"},
    "Overtime_Hours": {"min": 0.0, "max": 80.0, "default": 10.0, "step": 1.0, "unit": "hrs/mo"},
    "Previous_Performance_Score": {"min": 30.0, "max": 100.0, "default": 78.0, "step": 0.5, "unit": "score (0-100)"},
    "Years_Since_Last_Promotion": {"min": 0.0, "max": 20.0, "default": 2.0, "step": 0.5, "unit": "years"},
    "Team_Size": {"min": 1, "max": 50, "default": 8, "step": 1, "unit": "members"}
}


def score_to_category(score: float) -> str:
    """Classify continuous performance score into High, Medium, Low."""
    if score >= THRESHOLD_HIGH:
        return CATEGORY_HIGH
    elif score >= THRESHOLD_MEDIUM:
        return CATEGORY_MEDIUM
    else:
        return CATEGORY_LOW
