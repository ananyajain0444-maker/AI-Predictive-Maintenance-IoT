import joblib
import pandas as pd

def predict_sample(temp, vib, curr):
    model = joblib.load("models/model.pkl")

    # Create DataFrame with column names
    input_data = pd.DataFrame({
        "temperature": [temp],
        "vibration": [vib],
        "current": [curr]
    })

    result = model.predict(input_data)

    return "⚠️ Failure" if result[0]==1 else "✅ Normal"
