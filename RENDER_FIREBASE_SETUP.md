# Setting Firebase Key in Render - Step by Step

## The Problem
Render shows: "Firebase key not found - running in offline mode"
This means the `FIREBASE_KEY_JSON` environment variable is not set correctly.

## Solution

### Step 1: Get the JSON String
Run this command in the `ml-backend` directory:
```bash
python get-firebase-env-value.py
```

This will output a single-line JSON string. **Copy the entire string** (it's all on one line).

### Step 2: Set in Render Dashboard

1. Go to: https://dashboard.render.com
2. Click on your **`taskflow-ml-api`** service
3. Click **"Environment"** in the left sidebar
4. Look for existing environment variables
5. **If `FIREBASE_KEY_JSON` exists:**
   - Click on it to edit
   - Delete the old value
   - Paste the new JSON string (from Step 1)
   - Click **"Save Changes"**
6. **If `FIREBASE_KEY_JSON` doesn't exist:**
   - Click **"Add Environment Variable"** or **"Add Secret"**
   - **Key**: `FIREBASE_KEY_JSON` (exactly, case-sensitive)
   - **Value**: Paste the JSON string from Step 1
   - Click **"Save Changes"**

### Step 3: Important Notes

⚠️ **Common Mistakes:**
- ❌ Don't add quotes around the JSON string
- ❌ Don't add extra spaces at the start/end
- ❌ Don't break it into multiple lines
- ✅ Paste it exactly as one continuous string
- ✅ The value should start with `{"type":"service_account"...`

### Step 4: Redeploy

After saving:
1. Render will automatically redeploy (or click "Manual Deploy" → "Deploy latest commit")
2. Wait 2-3 minutes for deployment
3. Check the logs - you should see:
   ```
   Found FIREBASE_KEY_JSON environment variable (length: XXXX)
   Successfully parsed JSON from environment variable
   Firebase initialized successfully from environment variable
   ```

### Step 5: Verify

After deployment, test:
```bash
curl https://taskflowml.onrender.com/health
```

Check the logs - if you see "Firebase initialized successfully from environment variable", it's working!

## Troubleshooting

**Still seeing "Firebase key not found"?**

1. **Check the logs** - Look for:
   - "FIREBASE_KEY_JSON environment variable not found" → Variable not set
   - "Failed to parse JSON" → Invalid JSON format
   - Check the error message for details

2. **Verify the variable:**
   - Go to Render Dashboard → Environment
   - Make sure `FIREBASE_KEY_JSON` exists
   - Click on it to verify it starts with `{"type":"service_account"`

3. **Try removing and re-adding:**
   - Delete the `FIREBASE_KEY_JSON` variable
   - Add it again with the fresh JSON string
   - Save and redeploy

4. **Check for hidden characters:**
   - Make sure there are no extra quotes, spaces, or line breaks
   - The JSON should be a single continuous string

