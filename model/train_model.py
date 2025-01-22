from data.data_loader import read_data, scale
from sklearn.linear_model import LinearRegression
import pickle
from pathlib import Path

def load_scale_data(train_data_path, scaler_path):

    data=read_data(train_data_path)
    X_train=data.drop(columns=['Chance of Admit '])
    y_train=data['Chance of Admit ']
    X_train = scale(X_train, scaler_path)

    return X_train, y_train

def train_and_save_model(model_path, X_train, y_train):
    model=LinearRegression()
    model.fit(X_train, y_train)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model is trained successfully and saved at Path {model_path}")

if __name__=="__main__":
    X_train, y_train=load_scale_data(Path("Artifacts/data/train.csv"), 
                                     Path("Artifacts/scaler/scaler.pkl"))
    train_and_save_model(Path("Artifacts/model/model.pkl"), X_train, y_train)

