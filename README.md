# ML Backend for Task Scheduling Optimization

This Python Flask backend provides machine learning-powered task scheduling recommendations using a CART (Classification and Regression Tree) decision tree model.

## Features

- **CART Decision Tree Model**: Trained on survey data to predict optimal task scheduling times
- **Real-time Predictions**: RESTful API for getting scheduling recommendations
- **User Insights**: Personalized analytics based on task completion patterns
- **Firebase Integration**: Direct connection to Firestore for user data

## Setup

1. **Install Dependencies**
   ```bash
   cd ml-backend
   pip install -r requirements.txt
   ```

2. **Firebase Setup**
   - Download your Firebase service account key
   - Save it as `firebase-key.json` in the ml-backend directory

3. **Run the Server**
   ```bash
   python app.py
   ```

## API Endpoints

### GET /health
Check if the service is running and model is loaded.

### POST /train
Retrain the ML model with updated survey data.

### POST /predict
Get optimal scheduling recommendations for a task.

**Request Body:**
```json
{
  "category": "Work",
  "priority": 3,
  "estimatedDuration": "1 – 2 hours"
}
```

**Response:**
```json
{
  "optimal_time": "Evening (6 PM – 12 MN)",
  "confidence": 0.85,
  "features_used": {...}
}
```

### GET /insights/{user_id}
Get personalized insights for a user based on their task history.

## ML Model Details

The CART model is trained on survey responses that include:
- Task types (Academic, Work, Personal, etc.)
- Priority levels (High, Medium, Low)
- Estimated durations
- Deadline proximity preferences
- Rescheduling frequency
- Productive time preferences
- Reminder preferences

The model predicts the optimal time of day for scheduling tasks based on these features.

## Deployment

For production deployment, consider:
- **Google Cloud Run** for serverless deployment
- **Firebase Cloud Functions** for integrated Firebase deployment
- **Docker** for containerized deployment

## Model Training

The model is automatically trained when the server starts. To retrain with new data:

1. Update the CSV file
2. Call the `/train` endpoint
3. The model will be updated with new predictions