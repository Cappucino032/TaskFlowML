# ML API for Personal Activity Scheduler

This is the machine learning backend for the Personal Activity Scheduler React Native app, deployed on Railway.

## 🚀 Deployment Status

**Railway URL**: [To be updated after deployment]

## 📊 API Endpoints

### GET /health
Check service status and model loading.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /predict
Get optimal task scheduling recommendations.

**Request:**
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
Get personalized productivity insights.

### POST /train
Retrain the ML model (admin only).

## 🔧 Local Development

```bash
cd ml-api
pip install -r requirements.txt
python app.py
```

## 🚀 Railway Deployment

The API is configured for Railway deployment with:
- Automatic health checks
- Environment variable support
- Docker containerization
- Production logging

## 📱 Integration

Update your React Native app's ML service URL to point to the Railway deployment:

```typescript
// In src/services/mlService.ts
const API_BASE_URL = 'https://your-railway-app.up.railway.app';
```

## 🧪 Testing

```bash
# Health check
curl https://your-app.up.railway.app/health

# Test prediction
curl -X POST https://your-app.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"category": "Work", "priority": 3}'