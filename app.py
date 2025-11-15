from flask import Flask, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
import os
from datetime import datetime
import joblib
import json
import warnings

app = Flask(__name__)

# Initialize Firebase Admin SDK
# Supports both environment variable (production) and file path (local development)
try:
    # Option 1: Check for environment variable with JSON content (for production/Render)
    firebase_key_json = os.environ.get('FIREBASE_KEY_JSON')
    if firebase_key_json:
        print(f"Found FIREBASE_KEY_JSON environment variable (length: {len(firebase_key_json)})")
        print(f"First 50 chars: {firebase_key_json[:50]}")
        try:
            # Strip whitespace and newlines that Render might add
            firebase_key_json = firebase_key_json.strip()
            # Parse JSON from environment variable
            key_data = json.loads(firebase_key_json)
            print("Successfully parsed JSON from environment variable")
            # Verify it has required fields
            if 'type' not in key_data or key_data.get('type') != 'service_account':
                raise ValueError("Invalid service account key: missing or incorrect 'type' field")
            if 'private_key' not in key_data:
                raise ValueError("Invalid service account key: missing 'private_key' field")
            print("Service account key structure validated")
            cred = credentials.Certificate(key_data)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            firebase_enabled = True
            print("Firebase initialized successfully from environment variable")
        except json.JSONDecodeError as json_err:
            print(f"Failed to parse JSON from environment variable: {json_err}")
            print(f"First 100 chars of value: {firebase_key_json[:100]}")
            print(f"Last 50 chars of value: {firebase_key_json[-50:]}")
            raise
        except Exception as parse_err:
            print(f"Error processing Firebase key from environment variable: {parse_err}")
            raise
    else:
        print("FIREBASE_KEY_JSON environment variable not found")
        # Option 2: Fall back to file path (for local development)
        firebase_key_path = os.environ.get('FIREBASE_KEY_PATH', 'firebase-key.json')
        print(f"Checking for Firebase key file at: {firebase_key_path}")
        if os.path.exists(firebase_key_path):
            cred = credentials.Certificate(firebase_key_path)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            firebase_enabled = True
            print("Firebase initialized successfully from file")
        else:
            print(f"Firebase key file not found at {firebase_key_path} - running in offline mode")
            firebase_enabled = False
            db = None
except Exception as e:
    print(f"Firebase initialization failed: {e} - running in offline mode")
    import traceback
    traceback.print_exc()
    firebase_enabled = False
    db = None

# Global variables for ML model
model = None
label_encoders = {}
models_by_category = {}  # Store separate models for each category
feature_columns = ['task_type', 'priority_level', 'estimated_duration', 'deadline_proximity', 'reschedule_frequency', 'productive_time', 'preferred_time', 'reminder_preference']

# Model persistence paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
GENERAL_MODEL_PATH = os.path.join(MODELS_DIR, 'general_model.joblib')
GENERAL_ENCODERS_PATH = os.path.join(MODELS_DIR, 'general_encoders.joblib')
CATEGORY_MODELS_DIR = os.path.join(MODELS_DIR, 'categories')
os.makedirs(CATEGORY_MODELS_DIR, exist_ok=True)

def load_and_preprocess_data():
    """Load survey data and preprocess for ML training"""
    try:
        # Try multiple possible paths for the CSV file
        possible_paths = [
            '../Survey Questions for ML Training Data.csv',
            './data/Survey Questions for ML Training Data.csv',
            'Survey Questions for ML Training Data.csv',
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

def save_models():
    """Save trained models and encoders to disk for faster cold starts"""
    try:
        # Save general model and encoders
        if model is not None:
            joblib.dump(model, GENERAL_MODEL_PATH)
            joblib.dump(label_encoders, GENERAL_ENCODERS_PATH)
            print(f"Saved general model to {GENERAL_MODEL_PATH}")

        # Save category-specific models
        for category, category_data in models_by_category.items():
            # Sanitize category name for filename
            safe_category = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in category)
            category_model_path = os.path.join(CATEGORY_MODELS_DIR, f"{safe_category}_model.joblib")
            category_encoders_path = os.path.join(CATEGORY_MODELS_DIR, f"{safe_category}_encoders.joblib")
            
            joblib.dump(category_data['model'], category_model_path)
            joblib.dump(category_data['encoders'], category_encoders_path)
            print(f"Saved model for category '{category}' to {category_model_path}")

        return True
    except Exception as e:
        print(f"Error saving models: {e}")
        return False

def load_models():
    """Load trained models and encoders from disk if they exist"""
    global model, label_encoders, models_by_category
    
    try:
        loaded_any = False
        
        # Load general model
        if os.path.exists(GENERAL_MODEL_PATH) and os.path.exists(GENERAL_ENCODERS_PATH):
            model = joblib.load(GENERAL_MODEL_PATH)
            label_encoders = joblib.load(GENERAL_ENCODERS_PATH)
            print(f"Loaded general model from {GENERAL_MODEL_PATH}")
            loaded_any = True

        # Load category-specific models
        if os.path.exists(CATEGORY_MODELS_DIR):
            for filename in os.listdir(CATEGORY_MODELS_DIR):
                if filename.endswith('_model.joblib'):
                    category_name = filename.replace('_model.joblib', '')
                    encoders_filename = filename.replace('_model.joblib', '_encoders.joblib')
                    
                    model_path = os.path.join(CATEGORY_MODELS_DIR, filename)
                    encoders_path = os.path.join(CATEGORY_MODELS_DIR, encoders_filename)
                    
                    if os.path.exists(encoders_path):
                        category_model = joblib.load(model_path)
                        category_encoders = joblib.load(encoders_path)
                        
                        # Try to get accuracy from a metadata file, or default to 0.8
                        models_by_category[category_name] = {
                            'model': category_model,
                            'encoders': category_encoders,
                            'accuracy': 0.8  # Default accuracy if not saved
                        }
                        print(f"Loaded model for category '{category_name}' from {model_path}")
                        loaded_any = True

        return loaded_any
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def train_model():
    """Train separate CART decision tree models for each category"""
    global model, label_encoders, models_by_category

    df = load_and_preprocess_data()
    if df is None:
        return False

    # Get unique categories from the data
    categories = df['task_type'].unique()
    print(f"Training models for categories: {list(categories)}")

    models_by_category = {}

    for category in categories:
        print(f"Training model for category: {category}")

        # Filter data for this category
        category_df = df[df['task_type'] == category].copy()

        if len(category_df) < 5:  # Skip if too few samples
            print(f"Skipping {category} - insufficient data ({len(category_df)} samples)")
            continue

        # Encode categorical variables for this category
        category_encoders = {}
        encoded_df = category_df.copy()

        for column in feature_columns + ['optimal_schedule_time']:
            if column in encoded_df.columns:
                le = LabelEncoder()
                encoded_df[column] = le.fit_transform(encoded_df[column].astype(str))
                category_encoders[column] = le

        # Prepare features and target
        X = encoded_df[feature_columns]
        y = encoded_df['optimal_schedule_time']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train CART model (Decision Tree) for this category
        # Suppress sklearn feature name warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='sklearn')
            category_model = DecisionTreeClassifier(
                criterion='gini',
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            category_model.fit(X_train, y_train)

        # Calculate accuracy
        accuracy = category_model.score(X_test, y_test)
        print(f"Model for {category} trained with accuracy: {accuracy:.2f}")

        # Store model and encoders
        models_by_category[category] = {
            'model': category_model,
            'encoders': category_encoders,
            'accuracy': accuracy
        }

    # Keep a general model as fallback
    label_encoders = {}
    encoded_df = df.copy()

    for column in feature_columns + ['optimal_schedule_time']:
        if column in encoded_df.columns:
            le = LabelEncoder()
            encoded_df[column] = le.fit_transform(encoded_df[column].astype(str))
            label_encoders[column] = le

    X = encoded_df[feature_columns]
    y = encoded_df['optimal_schedule_time']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Suppress sklearn feature name warnings for general model
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module='sklearn')
        model = DecisionTreeClassifier(
            criterion='gini',
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"General model trained with accuracy: {accuracy:.2f}")

    # Save models to disk for faster cold starts
    save_models()

    return True

def predict_optimal_schedule(task_features):
    """Predict optimal scheduling time for a task using category-specific model"""
    if not models_by_category and model is None:
        return {"error": "Models not trained"}

    try:
        task_category = task_features.get('task_type', 'Personal')

        # Try to use category-specific model first
        if task_category in models_by_category:
            category_data = models_by_category[task_category]
            category_model = category_data['model']
            category_encoders = category_data['encoders']

            # Encode input features using category-specific encoders
            encoded_features = []
            for feature in feature_columns:
                value = task_features.get(feature, 'Unknown')
                if feature in category_encoders:
                    try:
                        encoded_value = category_encoders[feature].transform([str(value)])[0]
                    except:
                        # Handle unknown categories
                        encoded_value = 0
                    encoded_features.append(encoded_value)
                else:
                    encoded_features.append(0)

            # Make prediction with category-specific model
            # Suppress sklearn feature name warnings during prediction
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', module='sklearn')
                prediction = category_model.predict([encoded_features])[0]
                confidence = float(category_model.predict_proba([encoded_features]).max())

            # Decode prediction
            if 'optimal_schedule_time' in category_encoders:
                predicted_time = category_encoders['optimal_schedule_time'].inverse_transform([prediction])[0]
            else:
                predicted_time = "Unknown"

            return {
                "optimal_time": predicted_time,
                "confidence": confidence,
                "model_used": f"category-specific ({task_category})",
                "features_used": task_features
            }
        else:
            # Fall back to general model
            print(f"No specific model for category {task_category}, using general model")

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

            # Make prediction with general model
            # Suppress sklearn feature name warnings during prediction
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', module='sklearn')
                prediction = model.predict([encoded_features])[0]

            # Decode prediction
            if 'optimal_schedule_time' in label_encoders:
                predicted_time = label_encoders['optimal_schedule_time'].inverse_transform([prediction])[0]
            else:
                predicted_time = "Unknown"

            # Suppress sklearn warnings for predict_proba as well
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', module='sklearn')
                confidence = float(model.predict_proba([encoded_features]).max())

            return {
                "optimal_time": predicted_time,
                "confidence": confidence,
                "model_used": "general",
                "features_used": task_features
            }

    except Exception as e:
        return {"error": str(e)}

@app.route('/debug/firebase', methods=['GET'])
def debug_firebase():
    """Debug endpoint to check Firebase configuration status"""
    firebase_key_json = os.environ.get('FIREBASE_KEY_JSON')
    firebase_key_path = os.environ.get('FIREBASE_KEY_PATH', 'firebase-key.json')
    
    debug_info = {
        'firebase_enabled': firebase_enabled,
        'has_env_var': firebase_key_json is not None,
        'env_var_length': len(firebase_key_json) if firebase_key_json else 0,
        'env_var_starts_with': firebase_key_json[:50] if firebase_key_json else None,
        'has_file_path_env': os.environ.get('FIREBASE_KEY_PATH') is not None,
        'file_exists': os.path.exists(firebase_key_path) if firebase_key_path else False,
        'all_env_vars': [k for k in os.environ.keys() if 'FIREBASE' in k.upper()],
    }
    
    # Try to parse if env var exists
    if firebase_key_json:
        try:
            test_parse = json.loads(firebase_key_json)
            debug_info['json_parse_success'] = True
            debug_info['json_has_type'] = test_parse.get('type') == 'service_account'
        except Exception as e:
            debug_info['json_parse_success'] = False
            debug_info['json_parse_error'] = str(e)
    
    return jsonify(debug_info)

@app.route('/health', methods=['GET'])
def health_check():
    category_models_count = len(models_by_category) if models_by_category else 0
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "category_models": category_models_count,
        "categories": list(models_by_category.keys()) if models_by_category else []
    })

@app.route('/keepalive', methods=['GET', 'HEAD'])
def keepalive():
    """Keep-alive endpoint to prevent Render free tier from sleeping"""
    response = jsonify({
        "status": "awake",
        "message": "Service is active"
    })
    # Handle HEAD requests (used by monitoring services like UptimeRobot)
    if request.method == 'HEAD':
        return '', 200
    return response

@app.route('/train', methods=['POST'])
def train():
    success = train_model()
    return jsonify({"success": success, "message": "Model trained" if success else "Training failed"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_id = data.get('userId')

    # Try personalized prediction first if user has task history
    if user_id and firebase_enabled:
        try:
            personalized_result = predict_from_user_history(user_id, data)
            if personalized_result and 'error' not in personalized_result:
                return jsonify(personalized_result)
        except Exception as e:
            print(f"Personalized prediction failed: {e}")

    # Fall back to survey-based prediction
    task_features = {
        'task_type': data.get('category', 'Personal'),
        'priority_level': 'High' if data.get('priority') == 3 else 'Medium' if data.get('priority') == 2 else 'Low',
        'estimated_duration': data.get('estimatedDuration', '30 minutes – 1 hour'),
        'deadline_proximity': '1–2 days before',
        'reschedule_frequency': 'Rarely (1–2 times a week)',
        'productive_time': 'Evening (6 PM – 12 MN)',
        'preferred_time': 'Evening (6 PM – 12 MN)',
        'reminder_preference': 'Early reminders (1–2 days before)'
    }

    result = predict_optimal_schedule(task_features)
    return jsonify(result)

@app.route('/insights/<user_id>', methods=['GET'])
def get_user_insights(user_id):
    """Get personalized insights for a user based on their task history"""
    if not firebase_enabled:
        # Provide fallback insights when Firebase is not configured
        # Don't show error message to user, just provide helpful general insights
        fallback_insights = [
            "Based on general productivity patterns, morning hours (6 AM - 12 PM) tend to be most productive for most users.",
            "Breaking tasks into smaller, manageable chunks can improve completion rates.",
            "Setting specific start and end times for tasks helps maintain focus and productivity.",
            "Regular breaks between tasks can help maintain energy levels throughout the day.",
            "Prioritizing high-importance tasks earlier in the day can lead to better outcomes."
        ]
        return jsonify({
            "insights": fallback_insights,
            "total_tasks": 0
        })

    try:
        # Get user's tasks from Firestore (using filter keyword to avoid deprecation warning)
        tasks_ref = db.collection('tasks').where(filter=FieldFilter('userId', '==', user_id))
        tasks = tasks_ref.stream()

        task_data = []
        for task in tasks:
            task_dict = task.to_dict()
            task_dict['id'] = task.id
            task_data.append(task_dict)

        if not task_data:
            # Provide helpful insights even when user has no tasks yet
            return jsonify({
                "insights": [
                    "No task history available yet. Complete some tasks to see personalized insights!",
                    "Start by creating tasks with specific start and end times.",
                    "Mark tasks as complete to help the AI learn your productivity patterns.",
                    "Use the calendar view to visualize your task schedule."
                ],
                "total_tasks": 0
            })

        # Analyze patterns
        insights = analyze_user_patterns(task_data)

        return jsonify({"insights": insights, "total_tasks": len(task_data)})

    except Exception as e:
        # Provide fallback insights on error
        fallback_insights = [
            "Unable to load personalized insights at this time.",
            "General tip: Schedule tasks during your most productive hours.",
            "Break large tasks into smaller, manageable pieces.",
            "Set reminders to stay on track with your schedule."
        ]
        return jsonify({
            "insights": fallback_insights,
            "total_tasks": 0,
            "error": str(e)
        })

def predict_from_user_history(user_id, task_data):
    """Predict optimal schedule based on user's own completed task history"""
    try:
        # Get user's completed tasks (using filter keyword to avoid deprecation warning)
        tasks_ref = db.collection('tasks').where(filter=FieldFilter('userId', '==', user_id)).where(filter=FieldFilter('completed', '==', True))
        tasks = tasks_ref.stream()

        completed_tasks = []
        for task in tasks:
            task_dict = task.to_dict()
            task_dict['id'] = task.id
            completed_tasks.append(task_dict)

        if len(completed_tasks) < 3:
            return None  # Not enough data for personalized prediction

        # Get productivity logs for completion times (using filter keyword to avoid deprecation warning)
        productivity_ref = db.collection('productivityLogs').where(filter=FieldFilter('userId', '==', user_id))
        logs = productivity_ref.stream()

        completion_data = {}
        for log in logs:
            log_dict = log.to_dict()
            task_id = log_dict.get('taskId')
            if task_id:
                completion_data[task_id] = log_dict

        # Analyze patterns by category
        category_patterns = {}
        current_category = task_data.get('category', 'Personal')

        for task in completed_tasks:
            category = task.get('category', 'Personal')
            if category not in category_patterns:
                category_patterns[category] = []

            # Get completion time from productivity log or estimate
            task_id = task.get('id')
            log_data = completion_data.get(task_id, {})

            # Extract completion time (prefer actual log data, fallback to due date)
            completed_at = None
            if log_data.get('completedAt'):
                completed_at = log_data['completedAt']
            elif task.get('dueAt'):
                # Assume completed near due date if no log
                due_date = task['dueAt']
                if isinstance(due_date, str):
                    due_date = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
                completed_at = due_date

            if completed_at:
                completion_hour = completed_at.hour
                category_patterns[category].append({
                    'hour': completion_hour,
                    'priority': task.get('priority', 2),
                    'weekday': completed_at.weekday()
                })

        # Find best time for current category
        if current_category in category_patterns and category_patterns[current_category]:
            category_data = category_patterns[current_category]

            # Group by hour and count successes
            hour_counts = {}
            for entry in category_data:
                hour = entry['hour']
                hour_counts[hour] = hour_counts.get(hour, 0) + 1

            if hour_counts:
                best_hour = max(hour_counts, key=hour_counts.get)
                confidence = hour_counts[best_hour] / len(category_data)

                # Convert hour to time period
                if 6 <= best_hour < 12:
                    time_period = "Morning (6 AM – 12 NN)"
                elif 12 <= best_hour < 18:
                    time_period = "Afternoon (12 NN – 6 PM)"
                elif 18 <= best_hour < 24:
                    time_period = "Evening (6 PM – 12 MN)"
                else:
                    time_period = "Late Night (12 MN – 3 AM)"

                return {
                    "optimal_time": time_period,
                    "confidence": confidence,
                    "model_used": f"personalized ({current_category})",
                    "sample_size": len(category_data),
                    "reasoning": f"Based on {len(category_data)} completed {current_category.lower()} tasks"
                }

        return None

    except Exception as e:
        print(f"Error in personalized prediction: {e}")
        return None

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

# Initialize models when module is imported (works with both Flask dev server and Gunicorn)
# This ensures models are loaded even when using Gunicorn (which doesn't run __main__)
# Wrap in try-except to ensure app starts even if model loading fails
try:
    print("Initializing ML models...")
    models_loaded = load_models()

    if not models_loaded:
        # If no saved models exist, train new ones
        print("No saved models found. Training ML model...")
        train_model()
    else:
        print("Successfully loaded saved models! (Cold start optimized)")
except Exception as e:
    print(f"Warning: Model initialization failed: {e}")
    print("App will continue to run, but ML predictions may not work until models are trained.")
    # Set models to None so the app knows they're not loaded
    model = None
    models_by_category = {}

# Only run Flask dev server if script is executed directly (not when imported by Gunicorn)
if __name__ == '__main__':
    # Get port from environment variable (Railway/Render sets this)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)