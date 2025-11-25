# Render Free Tier ML Backend Fixes

## 🎯 **Issues Fixed**

### 1. **Timeout Issues - FIXED** ✅
**Problem**: ML service was timing out after 10-15 seconds, but Render free tier cold starts can take 30-60 seconds.

**Solution**: 
- Increased all ML service timeouts to **90 seconds** to handle cold starts
- Updated `PersonalActivitySchedulerTS/src/services/mlService.ts`:
  - Health check: 10s → 90s
  - User insights: 15s → 90s  
  - Predictions: No timeout → 90s

### 2. **Production WSGI Server - FIXED** ✅
**Problem**: Flask's development server is not suitable for production and is slower.

**Solution**:
- Updated `ml-backend/Dockerfile` to use **Gunicorn** (already in requirements.txt)
- Configured with 2 workers and 120s timeout for cold starts
- Command: `gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 app:app`

### 3. **Model Persistence - FIXED** ✅
**Problem**: Models were retrained on every cold start, taking 10-30 seconds.

**Solution**:
- Added `joblib` to `requirements.txt` for model serialization
- Models are now saved to disk after training (`/app/models/`)
- On startup, the app tries to load saved models first
- Only trains new models if saved models don't exist
- **Note**: Render free tier has ephemeral storage, so models will be retrained on first request after a restart, but subsequent requests in the same session will be fast.

### 4. **Keep-Alive Endpoint - ADDED** ✅
**Problem**: Render free tier services sleep after 15 minutes of inactivity.

**Solution**:
- Added `/keepalive` endpoint that can be pinged periodically
- You can set up a cron job or service (like UptimeRobot) to ping this endpoint every 10-14 minutes to keep the service awake

---

## 📋 **Understanding Render Free Tier Limitations**

### **How Render Free Tier Works:**
1. **Sleep After Inactivity**: Services sleep after ~15 minutes of no requests
2. **Cold Start Time**: When waking up, it takes 30-60 seconds to:
   - Start the container
   - Load dependencies
   - Train/load ML models
   - Start Gunicorn
3. **Ephemeral Storage**: Files saved to disk are lost when the service restarts or sleeps

### **What This Means:**
- ✅ **First request after sleep**: Will take 30-60 seconds (cold start + model training)
- ✅ **Subsequent requests**: Will be fast (models loaded in memory)
- ⚠️ **After 15 min inactivity**: Service sleeps again
- ⚠️ **Models on disk**: Lost on restart, but retrained quickly on first request

---

## 🔧 **How to Keep Service Awake (Optional)**

### **Option 1: Use UptimeRobot (Free)**
1. Go to https://uptimerobot.com
2. Create a new monitor:
   - Type: HTTP(s)
   - URL: `https://taskflowml.onrender.com/keepalive`
   - Interval: 10 minutes
3. This will ping your service every 10 minutes to keep it awake

### **Option 2: Use Cron-Job.org (Free)**
1. Go to https://cron-job.org
2. Create a new cron job:
   - URL: `https://taskflowml.onrender.com/keepalive`
   - Schedule: Every 10 minutes
3. This will ping your service to prevent sleep

### **Option 3: Upgrade Render Plan**
- Paid plans don't sleep and have persistent storage
- Models would persist across restarts

---

## 📊 **CSV vs Model - Explained**

### **The CSV File** (`Survey Questions for ML Training Data.csv`)
- **What it is**: Training data (survey responses)
- **Purpose**: Used to train the ML model
- **Contains**: Examples of task types, priorities, durations, and optimal scheduling times

### **The Trained Model** (saved as `.joblib` files)
- **What it is**: The actual ML model (Decision Tree Classifier)
- **Purpose**: Makes predictions based on new task data
- **Contains**: Learned patterns from the CSV training data
- **Location**: `/app/models/` directory (general_model.joblib, category-specific models)

### **How It Works:**
1. **Training Phase** (first time or when models don't exist):
   - Load CSV training data
   - Train Decision Tree models (one per category + one general)
   - Save models to disk as `.joblib` files
   - Takes 10-30 seconds

2. **Prediction Phase** (when user creates a task):
   - Load saved models from disk (if available)
   - Use models to predict optimal scheduling time
   - Takes <1 second

3. **On Render Free Tier**:
   - Models are saved to disk after training
   - But disk is ephemeral (lost on restart)
   - So first request after restart retrains models
   - Subsequent requests use loaded models (fast)

---

## ✅ **What's Changed**

### **Files Modified:**
1. `PersonalActivitySchedulerTS/src/services/mlService.ts`
   - Increased timeouts to 90 seconds
   - Better error messages for cold starts

2. `ml-backend/app.py`
   - Added model persistence (save/load)
   - Added `/keepalive` endpoint
   - Try to load models before training

3. `ml-backend/Dockerfile`
   - Updated to use Gunicorn instead of Flask dev server
   - Added models directory creation

4. `ml-backend/requirements.txt`
   - Added `joblib==1.4.2` for model serialization

---

## 🚀 **Next Steps**

1. **Deploy Updated Code**:
   - Push changes to your repository
   - Render will automatically redeploy
   - First request will train and save models

2. **Set Up Keep-Alive** (Optional but Recommended):
   - Use UptimeRobot or similar service
   - Ping `/keepalive` every 10 minutes
   - This prevents the service from sleeping

3. **Test the Service**:
   ```bash
   curl https://taskflowml.onrender.com/health
   curl https://taskflowml.onrender.com/keepalive
   ```

4. **Monitor Performance**:
   - First request after sleep: 30-60 seconds (expected)
   - Subsequent requests: <1 second (fast)
   - If you set up keep-alive, service should rarely sleep

---

## 💡 **Tips**

- **For Development**: The 90-second timeout should handle most cold starts
- **For Production**: Consider setting up keep-alive or upgrading to a paid plan
- **User Experience**: The app now shows better error messages if ML service is cold starting
- **Model Accuracy**: Models are retrained from the same CSV, so accuracy remains consistent

---

## ⚠️ **Important Notes**

1. **Render Free Tier Limitations**:
   - Services sleep after 15 minutes of inactivity
   - Cold starts take 30-60 seconds
   - Storage is ephemeral (lost on restart)

2. **Model Persistence**:
   - Models are saved to disk for faster subsequent requests
   - But lost on restart (ephemeral storage)
   - First request after restart will retrain (one-time delay)

3. **Keep-Alive**:
   - Prevents service from sleeping
   - Free services like UptimeRobot work well
   - Set interval to 10-14 minutes (before 15-min sleep threshold)

