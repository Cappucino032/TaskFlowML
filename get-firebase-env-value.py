#!/usr/bin/env python3
"""
Helper script to format Firebase key JSON for environment variable.
Run this to get the properly formatted string to paste into Render.
"""
import json
import sys

def main():
    try:
        with open('firebase-key.json', 'r') as f:
            key_data = json.load(f)
        
        # Convert back to JSON string (compact, single line)
        json_string = json.dumps(key_data, separators=(',', ':'))
        
        print("=" * 80)
        print("Copy the text below and paste it as the VALUE for FIREBASE_KEY_JSON:")
        print("=" * 80)
        print()
        print(json_string)
        print()
        print("=" * 80)
        print("Instructions:")
        print("1. In Render Dashboard → Your Service → Environment")
        print("2. Click 'Add Environment Variable'")
        print("3. KEY: FIREBASE_KEY_JSON")
        print("4. VALUE: Paste the JSON string above")
        print("5. Save and redeploy")
        print("=" * 80)
        
    except FileNotFoundError:
        print("Error: firebase-key.json not found in current directory")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in firebase-key.json: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

