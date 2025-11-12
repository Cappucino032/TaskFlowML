# Firebase Setup Guide

## ✅ Local Development (Already Done)

Your `firebase-key.json` file is in place and working. The app will automatically use it when running locally.

## 🚀 Production Setup (Render)

To enable Firebase in your Render deployment:

### Step 1: Get Your Firebase Key Content

1. Open `ml-backend/firebase-key.json` in a text editor
2. Copy the **entire file contents** (all 14 lines, including the curly braces)
3. Keep it ready to paste

### Step 2: Add Environment Variable in Render

1. Go to your Render dashboard: https://dashboard.render.com
2. Navigate to your `taskflow-ml-api` service
3. Click on **"Environment"** in the left sidebar
4. Click **"Add Environment Variable"**
5. Set:
   - **Key**: `FIREBASE_KEY_JSON`
   - **Value**: Paste the entire JSON content from `firebase-key.json` (as a single string)
6. Click **"Save Changes"**
7. Render will automatically redeploy your service

### Step 3: Verify It's Working

After deployment, check the logs. You should see:
```
Firebase initialized successfully from environment variable
```

Instead of:
```
Firebase key not found - running in offline mode
```

### Step 4: Test the API

Once deployed, test the insights endpoint:
```bash
curl https://taskflowml.onrender.com/insights/YOUR_USER_ID
```

You should now get personalized insights based on your actual task data instead of generic fallback insights.

## 🔒 Security Notes

- ✅ `firebase-key.json` is already in `.gitignore` - it won't be committed
- ✅ Environment variables in Render are encrypted at rest
- ✅ Never share your Firebase key publicly
- ✅ If you suspect the key is compromised, regenerate it in Firebase Console

## 🐛 Troubleshooting

**Problem**: Still seeing "Firebase key not found" in production logs

**Solution**: 
- Make sure the environment variable is named exactly `FIREBASE_KEY_JSON` (case-sensitive)
- Ensure the entire JSON content is pasted (including `{` and `}`)
- Check that there are no extra spaces or line breaks at the start/end
- Redeploy the service after adding the variable

**Problem**: "Firebase initialization failed" error

**Solution**:
- Verify the JSON is valid (use a JSON validator)
- Check that all required fields are present in the key file
- Ensure the service account has proper permissions in Firebase Console

