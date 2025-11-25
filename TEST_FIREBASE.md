# Testing Firebase Setup on Render

## Current Status
The API is still returning old error messages, which means Render needs to be redeployed with the latest code.

## Steps to Fix:

### 1. Verify Code is Pushed to GitHub
Make sure all changes are committed and pushed:
```bash
git status
git add .
git commit -m "Add Firebase environment variable support"
git push
```

### 2. Verify Environment Variable in Render
1. Go to Render Dashboard → Your Service (`taskflow-ml-api`)
2. Click **"Environment"** tab
3. Verify `FIREBASE_KEY_JSON` exists
4. Click on it to verify the value starts with `{"type":"service_account"...`

### 3. Trigger Redeploy
Render should auto-deploy, but you can manually trigger:
1. Go to Render Dashboard → Your Service
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

### 4. Check Deployment Logs
After redeploy, check the logs. You should see:
```
Firebase initialized successfully from environment variable
```

If you see:
```
Firebase key not found - running in offline mode
```
Then the environment variable isn't being read correctly.

### 5. Test After Redeploy
```bash
# Test health
curl https://taskflowml.onrender.com/health

# Test insights (should return personalized or "no tasks" message, not error)
curl https://taskflowml.onrender.com/insights/YOUR_USER_ID
```

## Troubleshooting

**If still seeing "Firebase not configured":**
- Wait 2-3 minutes for deployment to complete
- Check Render logs for initialization messages
- Verify environment variable has no extra quotes or spaces
- Try removing and re-adding the environment variable

**If seeing initialization errors:**
- Check that the JSON is valid (no missing quotes, proper escaping)
- Verify all required fields are present in the key file
- Check Render logs for specific error messages

