import pandas as pd
import joblib
import os

def save_model(model, path='model/model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def save_predictions(y_test, y_pred, path='results/predictions.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    output.to_csv(path, index=False)
    print(f"Predictions saved to {path}")
