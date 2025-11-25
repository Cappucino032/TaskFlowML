# Firebase Key Troubleshooting Guide

## Current Issue
Render shows: "Firebase key not found - running in offline mode"

## Step-by-Step Fix

### 1. Verify Environment Variable is Set in Render

1. Go to: https://dashboard.render.com
2. Click on `taskflow-ml-api` service
3. Click **"Environment"** tab
4. Look for `FIREBASE_KEY_JSON` in the list
5. **If it's NOT there:**
   - Click **"+ Add"** → **"Secret"** (or "Environment Variable")
   - Key: `FIREBASE_KEY_JSON`
   - Value: Paste the JSON string from `firebase-env-value.txt`
   - Click **"Save, rebuild, and deploy"**

### 2. Important: After Adding/Editing, You MUST Redeploy

Render doesn't always auto-redeploy when you change environment variables. You need to:

1. After saving the environment variable, click **"Manual Deploy"** → **"Deploy latest commit"**
2. OR click **"Save, rebuild, and deploy"** button (if available)

### 3. Check Deployment Logs

After deployment starts, check the logs. Look for these messages:

**✅ SUCCESS - You should see:**
```
Found FIREBASE_KEY_JSON environment variable (length: XXXX)
First 50 chars: {"type":"service_account","project_id":"productivity
Successfully parsed JSON from environment variable
Service account key structure validated
Firebase initialized successfully from environment variable
```

**❌ FAILURE - If you see:**
```
FIREBASE_KEY_JSON environment variable not found
```
→ The environment variable isn't set or isn't being read

**❌ FAILURE - If you see:**
```
Failed to parse JSON from environment variable: ...
```
→ The JSON format is wrong (check for extra quotes, spaces, etc.)

### 4. Test with Debug Endpoint

After deployment, test:
```bash
curl https://taskflowml.onrender.com/debug/firebase
```

This will show:
- Whether the environment variable exists
- Its length
- Whether JSON parsing works
- Current Firebase status

### 5. Common Issues and Fixes

#### Issue: "Environment variable not found"
**Fix:**
- Make sure variable name is exactly `FIREBASE_KEY_JSON` (case-sensitive)
- Make sure you clicked "Save" after adding it
- Make sure you triggered a redeploy after saving

#### Issue: "Failed to parse JSON"
**Fix:**
- The JSON might have extra quotes or formatting
- Delete the variable and add it again
- Copy the exact string from `firebase-env-value.txt` (line 1)
- Make sure it starts with `{` and ends with `}`

#### Issue: Variable exists but still "not found"
**Fix:**
- Render might need a full service restart
- Try: Render Dashboard → Your Service → Settings → "Restart Service"
- Or delete and recreate the environment variable

### 6. Alternative: Use Secret Files (If Environment Variable Doesn't Work)

If environment variable still doesn't work, try using Render's "Secret Files":

1. In Render Dashboard → Your Service → Environment
2. Scroll to "Secret Files" section
3. Click "+ Add"
4. Filename: `firebase-key.json`
5. Content: Paste the JSON (can be formatted with line breaks)
6. Save and redeploy

Then update the code to use:
```python
firebase_key_path = '/etc/secrets/firebase-key.json'
```

But first, try the environment variable approach - it should work!

## Quick Checklist

- [ ] Environment variable `FIREBASE_KEY_JSON` exists in Render
- [ ] Value is the complete JSON string (starts with `{`, ends with `}`)
- [ ] Clicked "Save" after adding/editing
- [ ] Triggered a redeploy (manual or auto)
- [ ] Checked deployment logs for initialization messages
- [ ] Tested `/debug/firebase` endpoint

