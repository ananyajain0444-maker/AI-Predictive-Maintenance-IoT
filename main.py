print("🔧 AI Predictive Maintenance System")
print("👩‍💻 Developed by Ananya Jain\n")

from src.preprocess import load_data
from src.train import train_model
from src.predict import predict_sample

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# ✅ Create folders
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ✅ STEP 1: LOAD DATA (VERY IMPORTANT - FIRST)
print("📂 Loading dataset...")
data = load_data("data/iot_sensor_data.csv")

# ✅ STEP 2: TRAIN MODEL
print("🤖 Training model...")
model, X_test, y_test = train_model(data)

# ✅ STEP 3: GENERATE GRAPHS
print("📊 Generating graphs...")

# 1️⃣ Temperature Trend
plt.figure()
plt.plot(data['temperature'])
plt.title("Temperature Trend")
plt.savefig("images/temp.png")
plt.close()

# 2️⃣ Failure Distribution
plt.figure()
plt.scatter(data['temperature'], data['failure'])
plt.title("Failure Distribution")
plt.savefig("images/failure_plot.png")
plt.close()

# 3️⃣ Prediction vs Actual
predictions = model.predict(X_test)

plt.figure()
plt.plot(predictions, label="Predicted")
plt.plot(y_test.values, label="Actual")
plt.legend()
plt.title("Prediction vs Actual")
plt.savefig("images/prediction_vs_actual.png")
plt.close()

# 4️⃣ Confusion Matrix
cm = confusion_matrix(y_test, predictions)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png")
plt.close()

# 5️⃣ Feature Importance
importance = model.feature_importances_

plt.figure()
plt.bar(['temperature','vibration','current'], importance)
plt.title("Feature Importance")
plt.savefig("images/feature_importance.png")
plt.close()

# ✅ STEP 4: TEST PREDICTION
print("🔍 Testing prediction...")
result = predict_sample(65, 4.2, 13)
print("Prediction:", result)

print("\n✅ DONE SUCCESSFULLY!")