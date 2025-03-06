#KartikBhaba_1178819
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

#Mod Load
model = joblib.load("heart_disease_model.pkl")


model_name = "Best Model according to anaysis"  

# Data load
file_path = "/Users/kartikbhaba/Desktop/WebHealthProject/heart_disease_uci.csv"  # Replace with your dataset's path
df = pd.read_csv(file_path)

# remove useless dat
df = df.drop(columns=["dataset"], errors='ignore')


# fbs encode - impute bad val
df["fbs"] = df["fbs"].map({True: 1, False: 0, 'True': 1, 'False': 0})  
imputer = SimpleImputer(strategy="most_frequent")
df["fbs"] = imputer.fit_transform(df[["fbs"]].values.reshape(-1, 1))  # Fill missing values in 'fbs'

# Gender Hand encode - impute bad val
df["sex"] = df["sex"].map({'Male': 1, 'Female': 0})  # Convert 'Male'/'Female' to 1/0
df["sex"] = imputer.fit_transform(df[["sex"]].values.reshape(-1, 1))  # Fill missing values in 'sex'

# 1-ht encode val
df = pd.get_dummies(df, columns=["cp", "restecg", "slope", "thal"], drop_first=True)

#median val fill
df.fillna(df.median(), inplace=True)

# split into x-featurs and y -target
X = df.drop(columns=["num"])  #num=target col
y = df["num"]

# Feat standardize (scaler=train)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale the input features

#pred mod use
predictions = model.predict(X_scaled)

# Mod name 
df["Model"] = model_name  # Mod type ->df save

# pred + csv to file //Question prob
df["Predictions"] = predictions
df.to_csv("predictions_output.csv", index=False)
print(f"Predictions saved to 'predictions_output.csv' using {model_name}")
