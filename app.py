from flask import Flask, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import os
from datetime import datetime

app = Flask(__name__)

# Initialize Firebase Admin SDK
# Note: You'll need to add your Firebase service account key as 'firebase-key.json'
try:
    # Check for Railway environment variable or local file
    firebase_key_path = os.environ.get('FIREBASE_KEY_PATH', 'firebase-key.json')
    if os.path.exists(firebase_key_path):
        cred = credentials.Certificate(firebase_key_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        firebase_enabled = True
        print("Firebase initialized successfully")
    else:
        print("Firebase key not found - running in offline mode")
        firebase_enabled = False
        db = None
except Exception as e:
    print(f"Firebase initialization failed: {e} - running in offline mode")
    firebase_enabled = False
    db = None

# Global variables for ML model
model = None
label_encoders = {}
feature_columns = ['task_type', 'priority_level', 'estimated_duration', 'deadline_proximity', 'reschedule_frequency', 'productive_time', 'preferred_time', 'reminder_preference']

def load_and_preprocess_data():
    """Load survey data and preprocess for ML training"""
    try:
        # Try multiple possible paths for the CSV file
        possible_paths = [
            './data/Survey Questions for ML Training Data.csv',
            'Survey Questions for ML Training Data.csv',
            '../Survey Questions for ML Training Data.csv',
            os.path.join(os.path.dirname(__file__), 'data', 'Survey Questions for ML Training Data.csv'),
            os.path.join(os.path.dirname(__file__), '..', 'Survey Questions for ML Training Data.csv')
        ]

        df = None
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                print(f"Successfully loaded data from: {path}")
                break
            except FileNotFoundError:
                continue

        if df is None:
            raise FileNotFoundError("Could not find survey data CSV file")

        # Rename columns for easier processing
        column_mapping = {
            'What type of tasks do you usually work on? ': 'task_type',
            'How do you usually rate the priority of your tasks? ': 'priority_level',
            'On average, how long does it take you to finish one task? ': 'estimated_duration',
            'How close to the deadline do you usually start working on a task?': 'deadline_proximity',
            'How often do you reschedule or postpone tasks? ': 'reschedule_frequency',
            'What time of day are you most productive? ': 'productive_time',
            'If you had a choice, when would you prefer to do your tasks? ': 'preferred_time',
            'When reminded about tasks, which option best helps you complete them? ': 'reminder_preference'
        }

        df = df.rename(columns=column_mapping)

        # Create target variable: optimal scheduling time based on productive_time and preferred_time
        def determine_optimal_time(row):
            productive = row['productive_time']
            preferred = row['preferred_time']

            # Priority: productive_time > preferred_time
            if productive == preferred:
                return productive
            elif productive in ['Morning (6 AM – 12 NN)', 'Afternoon (12 NN – 6 PM)', 'Evening (6 PM – 12 MN)', 'Late Night (12 MN – 3 AM)']:
                return productive
            else:
                return preferred

        df['optimal_schedule_time'] = df.apply(determine_optimal_time, axis=1)

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_model():
    """Train the CART decision tree model"""
    global model, label_encoders

    df = load_and_preprocess_data()
    if df is None:
        return False

    # Encode categorical variables
    label_encoders = {}
    encoded_df = df.copy()

    for column in feature_columns + ['optimal_schedule_time']:
        if column in encoded_df.columns:
            le = LabelEncoder()
            encoded_df[column] = le.fit_transform(encoded_df[column].astype(str))
            label_encoders[column] = le

    # Prepare features and target
    X = encoded_df[feature_columns]
    y = encoded_df['optimal_schedule_time']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train CART model (Decision Tree)
    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Calculate accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model trained with accuracy: {accuracy:.2f}")

    return True

def predict_optimal_schedule(task_features):
    """Predict optimal scheduling time for a task"""
    if model is None:
        return {"error": "Model not trained"}

    try:
        # Encode input features
        encoded_features = []
        for feature in feature_columns:
            value = task_features.get(feature, 'Unknown')
            if feature in label_encoders:
                try:
                    encoded_value = label_encoders[feature].transform([str(value)])[0]
                except:
                    # Handle unknown categories
                    encoded_value = 0
                encoded_features.append(encoded_value)
            else:
                encoded_features.append(0)

        # Make prediction
        prediction = model.predict([encoded_features])[0]

        # Decode prediction
        if 'optimal_schedule_time' in label_encoders:
            predicted_time = label_encoders['optimal_schedule_time'].inverse_transform([prediction])[0]
        else:
            predicted_time = "Unknown"

        return {
            "optimal_time": predicted_time,
            "confidence": float(model.predict_proba([encoded_features]).max()),
            "features_used": task_features
        }

    except Exception as e:
        return {"error": str(e)}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/train', methods=['POST'])
def train():
    success = train_model()
    return jsonify({"success": success, "message": "Model trained" if success else "Training failed"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract task features from request
    task_features = {
        'task_type': data.get('category', 'Personal'),
        'priority_level': 'High' if data.get('priority') == 3 else 'Medium' if data.get('priority') == 2 else 'Low',
        'estimated_duration': data.get('estimatedDuration', '30 minutes – 1 hour'),
        'deadline_proximity': '1–2 days before',  # Default assumption
        'reschedule_frequency': 'Rarely (1–2 times a week)',  # Default assumption
        'productive_time': 'Evening (6 PM – 12 MN)',  # Default assumption
        'preferred_time': 'Evening (6 PM – 12 MN)',  # Default assumption
        'reminder_preference': 'Early reminders (1–2 days before)'  # Default assumption
    }

    result = predict_optimal_schedule(task_features)
    return jsonify(result)

@app.route('/insights/<user_id>', methods=['GET'])
def get_user_insights(user_id):
    """Get personalized insights for a user based on their task history"""
    if not firebase_enabled:
        return jsonify({"insights": ["Firebase not configured - cannot access user data"]})

    try:
        # Get user's tasks from Firestore
        tasks_ref = db.collection('tasks').where('userId', '==', user_id)
        tasks = tasks_ref.stream()

        task_data = []
        for task in tasks:
            task_dict = task.to_dict()
            task_dict['id'] = task.id
            task_data.append(task_dict)

        if not task_data:
            return jsonify({"insights": ["No task history available yet. Complete some tasks to see insights!"]})

        # Analyze patterns
        insights = analyze_user_patterns(task_data)

        return jsonify({"insights": insights, "total_tasks": len(task_data)})

    except Exception as e:
        return jsonify({"error": str(e)})

def analyze_user_patterns(tasks):
    """Analyze user task patterns and provide insights"""
    insights = []

    # Completion rate
    completed_tasks = [t for t in tasks if t.get('completed', False)]
    completion_rate = len(completed_tasks) / len(tasks) if tasks else 0
    insights.append(f"Task completion rate: {completion_rate:.1%}")

    # Priority analysis
    priority_counts = {}
    for task in tasks:
        priority = task.get('priority', 2)
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    if priority_counts:
        most_common_priority = max(priority_counts, key=priority_counts.get)
        priority_names = {1: 'Low', 2: 'Medium', 3: 'High'}
        insights.append(f"You most frequently create {priority_names.get(most_common_priority, 'Medium')} priority tasks")

    # Category analysis
    category_counts = {}
    for task in tasks:
        category = task.get('category', 'Personal')
        category_counts[category] = category_counts.get(category, 0) + 1

    if category_counts:
        most_common_category = max(category_counts, key=category_counts.get)
        insights.append(f"Your most common task category is {most_common_category}")

    return insights

if __name__ == '__main__':
    # Train model on startup
    print("Training ML model...")
    train_model()

    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get('PORT', 5000))

    app.run(debug=False, host='0.0.0.0', port=port)