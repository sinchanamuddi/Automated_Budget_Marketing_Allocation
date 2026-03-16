import requests
import pandas as pd
import os

# 1. Configuration
API_URL = "http://localhost:5000/api/raw-sales" # Your Express API
SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Ecommerce.csv')

def sync_data():
    print("📡 Connecting to MERN API...")
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            # 2. Convert JSON to CSV
            df = pd.DataFrame(response.json())
            df.to_csv(SAVE_PATH, index=False)
            print(f"✅ Synced {len(df)} rows to {SAVE_PATH}")
        else:
            print(f"❌ API Error: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    sync_data()