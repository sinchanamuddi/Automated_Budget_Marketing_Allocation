import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Get the absolute path of the directory where train.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point to your new filename 'Ecommerce.csv'
data_path = os.path.join(BASE_DIR, '..', 'data', 'Ecommerce.csv')

def train_model():
    print(f"🔍 Searching for: {os.path.abspath(data_path)}")

    # Check if the file exists
    if not os.path.exists(data_path):
        print(f"❌ ERROR: File 'Ecommerce.csv' not found in E:\\AI_Budget_Optimizer\\data")
        print("Please check if the file extension is .csv and not .csv.csv")
        return

    # 1. Load the data
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded successfully! Columns: {list(df.columns)}")

    # 2. Preprocessing & Feature Engineering
    # We use lowercase 'revenue' and 'visit_date' based on your previous output
    df['Date'] = pd.to_datetime(df['visit_date'])
    df['is_weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Calculate spend (Cost)
    df['spend'] = df['unit_price'] * df['quantity']

    # Convert platform names to numbers
    le = LabelEncoder()
    df['channel_enc'] = le.fit_transform(df['marketing_channel'])

    # 3. Define Features and Target
    X = df[['channel_enc', 'spend', 'is_weekend']]
    y = df['revenue']

    # 4. Split and Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🧠 AI is analyzing your Indian E-commerce data...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Create 'models' folder if it doesn't exist
    model_dir = os.path.join(BASE_DIR, '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
        
    # 6. Save the results
    joblib.dump(model, os.path.join(model_dir, 'budget_model.pkl'))
    joblib.dump(le, os.path.join(model_dir, 'channel_encoder.pkl'))

    print("-" * 30)
    print(f"✅ SUCCESS! Model Accuracy: {model.score(X_test, y_test):.2f}")
    print(f"💾 'budget_model.pkl' is now ready in your models folder.")

if __name__ == "__main__":
    train_model()