# Heart Disease Prediction API

A FastAPI-based REST API that predicts heart disease risk using a trained SVM model. Send patient health features via a POST request and receive a prediction with a confidence level.

## Endpoints

| Method | Route        | Description                              |
|--------|--------------|------------------------------------------|
| GET    | `/`          | Welcome message and available endpoints  |
| GET    | `/features`  | List of expected feature names           |
| POST   | `/predict`   | Submit features and get a prediction     |
| GET    | `/docs`      | Interactive Swagger UI documentation     |

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python -m uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`

## Usage

### Get Required Features

```
GET /features
```

**Response:**
```json
{
  "feature_columns": [
    "high_blood_pressure", "high_cholesterol", "cholesterol_check",
    "bmi", "is_smoker", "had_stroke", "diabetes_status",
    "physical_activity", "eats_vegetables", "no_doctor_due_to_cost",
    "general_health", "mental_health_days", "physical_health_days",
    "difficulty_walking", "gender", "age_group",
    "education_level", "income_level"
  ],
  "count": 18
}
```

### Make a Prediction

```
POST /predict
```

**Request Body (dict format):**
```json
{
  "features": {
    "high_blood_pressure": 1,
    "high_cholesterol": 1,
    "cholesterol_check": 1,
    "bmi": 38.5,
    "is_smoker": 1,
    "had_stroke": 1,
    "diabetes_status": 2,
    "physical_activity": 0,
    "eats_vegetables": 0,
    "no_doctor_due_to_cost": 1,
    "general_health": 5,
    "mental_health_days": 20,
    "physical_health_days": 25,
    "difficulty_walking": 1,
    "gender": 0,
    "age_group": 12,
    "education_level": 2,
    "income_level": 2
  }
}
```

**Request Body (list format):**
```json
{
  "features": [1, 1, 1, 38.5, 1, 1, 2, 0, 0, 1, 5, 20, 25, 1, 0, 12, 2, 2]
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Heart Disease",
  "confidence": "82.31%",
  "probabilities": {
    "no_heart_disease": 17.69,
    "heart_disease": 82.31
  }
}
```

## Feature Descriptions

| Feature                 | Type    | Description                                      |
|-------------------------|---------|--------------------------------------------------|
| `high_blood_pressure`   | 0 or 1  | Has high blood pressure                         |
| `high_cholesterol`      | 0 or 1  | Has high cholesterol                            |
| `cholesterol_check`     | 0 or 1  | Cholesterol check in last 5 years               |
| `bmi`                   | float   | Body Mass Index                                  |
| `is_smoker`             | 0 or 1  | Has smoked at least 100 cigarettes in lifetime   |
| `had_stroke`            | 0 or 1  | Ever had a stroke                                |
| `diabetes_status`       | 0, 1, 2 | 0 = No, 1 = Pre-diabetes, 2 = Diabetes         |
| `physical_activity`     | 0 or 1  | Physical activity in past 30 days                |
| `eats_vegetables`       | 0 or 1  | Consumes vegetables 1+ times per day             |
| `no_doctor_due_to_cost` | 0 or 1  | Couldn't see doctor due to cost                  |
| `general_health`        | 1–5     | 1 = Excellent, 5 = Poor                         |
| `mental_health_days`    | 0–30    | Days of poor mental health in past 30 days       |
| `physical_health_days`  | 0–30    | Days of poor physical health in past 30 days     |
| `difficulty_walking`    | 0 or 1  | Serious difficulty walking or climbing stairs    |
| `gender`                | 0 or 1  | 0 = Female, 1 = Male                            |
| `age_group`             | 1–13    | Age category (1 = 18-24, 13 = 80+)              |
| `education_level`       | 1–6     | 1 = No school, 6 = College graduate              |
| `income_level`          | 1–8     | 1 = < $10k, 8 = $75k+                           |

## Deployment

The project includes a `Procfile` for deployment on platforms like Heroku or Render:

```
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

## Tech Stack

- **FastAPI** — Web framework
- **scikit-learn** — SVM model
- **imbalanced-learn** — Data balancing (used in training pipeline)
- **pandas / numpy** — Data handling
- **joblib** — Model serialization
