import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def read_data(data_path):
    data = pd.read_csv(data_path)
    return data

def feature_engineering(data):
    data = data.drop(columns=['Serial No.', 'SOP', 'University Rating'])
    return data

def split_data(data, train_data_path, test_data_path):
    train, test = train_test_split(data, test_size=0.2, shuffle=True)
    train = pd.DataFrame(train, columns=data.columns)
    train.to_csv(train_data_path, index=False)
    test = pd.DataFrame(test, columns=data.columns)
    test.to_csv(test_data_path, index=False)
    print(f"Train and Test data is saved at {train_data_path}, {test_data_path} respectively")


def scale(data, scaler_path):
    sc=StandardScaler()
    data=sc.fit_transform(data)
    with open(scaler_path,"wb") as f:
        pickle.dump(sc, f)
    return data

if __name__=="__main__":
    data=read_data(Path("Artifacts/data/jamboree.csv"))
    data = feature_engineering(data)
    split_data(data, Path("Artifacts/data/train.csv"), Path("Artifacts/data/test.csv"))

    




    
    
