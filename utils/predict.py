import pickle
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def load_scaler_and_model(model_path,scaler_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

def predict(input):
    model, scaler = load_scaler_and_model(Path("Artifacts/model/model.pkl"), Path("Artifacts/scaler/scaler.pkl"))
    scaled_input = scaler.transform(input)
    prediction = model.predict(scaled_input)
    return prediction

if __name__ == "__main__":
    data = np.array([[325,112,4.5,9.17,1]])
    prediction=predict(data)
    print(prediction)

