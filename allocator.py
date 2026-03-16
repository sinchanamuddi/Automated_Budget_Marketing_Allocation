import pandas as pd
import joblib
import os
import numpy as np

# Absolute paths for reliability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'budget_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, '..', 'models', 'channel_encoder.pkl')

def allocate_budget(total_budget, is_weekend):
    # 1. Load the trained "Brain"
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model not found. Please run train.py first.")
        return

    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)

    # 2. Identify the channels we know
    channels = le.classes_
    n_channels = len(channels)
    
    base_share = total_budget / n_channels
    results = []

    print(f"\n--- Budget Allocation for {'Weekend' if is_weekend else 'Weekday'} ---")
    print(f"Target Total Budget: ₹{total_budget:,}")
    
    for channel in channels:
        # Encode the channel name
        channel_idx = le.transform([channel])[0]
        
        # --- FIX START: Wrap input in a DataFrame with names ---
        input_data = pd.DataFrame(
            [[channel_idx, base_share, is_weekend]], 
            columns=['channel_enc', 'spend', 'is_weekend']
        )
        prediction = model.predict(input_data)[0]
        # --- FIX END ---
        
        results.append({
            'Channel': channel,
            'Recommended_Spend': base_share,
            'Predicted_Revenue': prediction,
            'ROI': prediction / base_share if base_share > 0 else 0
        })

    # 4. Show the Results
    alloc_df = pd.DataFrame(results)
    alloc_df = alloc_df.sort_values(by='ROI', ascending=False)
    
    print("\n", alloc_df[['Channel', 'Recommended_Spend', 'Predicted_Revenue', 'ROI']])
    print("\n✅ Strategy: Invest more in channels with high ROI.")