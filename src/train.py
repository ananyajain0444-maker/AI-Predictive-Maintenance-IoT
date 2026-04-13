from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(data):
    X = data[['temperature','vibration','current']]
    y = data['failure']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")
    return model, X_test, y_test