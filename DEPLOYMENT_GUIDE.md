# Railway Deployment Guide for ML Backend

## 🚀 Quick Deploy to Railway

### Step 1: Prepare Your Repository
1. **Initialize Git** (if not already done):
   ```bash
   cd ml-backend
   git init
   git add .
   git commit -m "Initial ML backend deployment"
   ```

2. **Create Railway Account**:
   - Go to [Railway.app](https://railway.app)
   - Sign up/Sign in with GitHub

### Step 2: Deploy to Railway
1. **Connect Repository**:
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository containing the `ml-backend` folder

2. **Configure Environment**:
   - Railway will auto-detect Python/Flask
   - Set environment variables in Railway dashboard:
     ```
     PORT=5000
     FIREBASE_KEY_PATH=/app/firebase-key.json  # Optional
     ```

### Step 3: Upload Firebase Key (Optional)
If you want Firebase integration:

1. **Download Firebase Service Account Key**:
   - Go to Firebase Console → Project Settings → Service Accounts
   - Generate new private key → Download JSON

2. **Upload to Railway**:
   - In Railway project → Variables → Add `FIREBASE_KEY_PATH`
   - Set value to `/app/firebase-key.json`
   - Upload the JSON file to Railway's file storage

### Step 4: Get Your API URL
- After deployment, Railway provides a URL like: `https://your-app-name.up.railway.app`
- Your API endpoints will be:
  - Health check: `https://your-app-name.up.railway.app/health`
  - Predictions: `https://your-app-name.up.railway.app/predict`
  - User insights: `https://your-app-name.up.railway.app/insights/{user_id}`

## 🔧 Configuration Files Created

### `Dockerfile`
- Multi-stage build for optimal size
- Includes all Python dependencies
- Health check endpoint configured

### `railway.json`
- Railway-specific deployment configuration
- Health checks and restart policies
- Nixpacks builder specification

### `.gitignore`
- Excludes sensitive files (Firebase keys)
- Python cache and virtual environments
- IDE and OS-specific files

## 📊 Testing Your Deployment

### Health Check
```bash
curl https://your-app-name.up.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Test Prediction
```bash
curl -X POST https://your-app-name.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category": "Work",
    "priority": 3,
    "estimatedDuration": "1 – 2 hours"
  }'
```

Expected response:
```json
{
  "optimal_time": "Evening (6 PM – 12 MN)",
  "confidence": 0.85,
  "features_used": {
    "category": "Work",
    "priority": 3,
    "estimatedDuration": "1 – 2 hours"
  }
}
```

## 🔄 Updating Your Deployment

1. **Make changes** to your code
2. **Commit and push** to your repository:
   ```bash
   git add .
   git commit -m "Update ML model"
   git push origin main
   ```
3. **Railway auto-deploys** the changes

## 💰 Railway Pricing

- **Free Tier**: 512MB RAM, 1GB storage, 100 hours/month
- **Hobby Plan**: $5/month - 1GB RAM, 5GB storage, unlimited hours
- **Pro Plan**: $10/month - 2GB RAM, 10GB storage, unlimited hours

## 🐛 Troubleshooting

### Common Issues:

1. **Model not loading**:
   - Check if CSV file is in the correct path
   - Verify Railway logs for file path errors

2. **Firebase connection failed**:
   - Ensure Firebase key is uploaded correctly
   - Check `FIREBASE_KEY_PATH` environment variable

3. **Port issues**:
   - Railway automatically assigns ports
   - Don't hardcode ports in your code

4. **Memory issues**:
   - Upgrade to Hobby plan if you hit memory limits
   - Optimize your ML model size

### Logs and Debugging:
- Check Railway dashboard → Deployments → View Logs
- Use the health endpoint to verify service status

## 🎯 Next Steps

1. **Update your React Native app** to use the Railway URL
2. **Test all endpoints** thoroughly
3. **Monitor performance** in Railway dashboard
4. **Set up monitoring** for production reliability

Your ML backend is now ready for production! 🚀