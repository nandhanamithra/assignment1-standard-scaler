import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

#readng the file
data = pd.read_csv(r"C:\flutter_projects\INTERNSHIP\lvl 2\assignment\insurance.csv")

#encoder
sex_encoder=LabelEncoder()
smoker_encoder=LabelEncoder()
region_encoder=LabelEncoder()
data["sex"]=sex_encoder.fit_transform(data["sex"]) 
data["smoker"]=smoker_encoder.fit_transform(data["smoker"])
data["region"]=region_encoder.fit_transform(data["region"])

#input abd target variables 
x= data.drop("charges", axis=1)
y = data["charges"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#standard scaler for input features
x_scaler = StandardScaler()

#test and train
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)

model = LinearRegression() 
model.fit(x_train, y_train)

#prediction
y_pred = model.predict(x_test)

#metrics for model preformance
error = mean_squared_error(y_test, y_pred)
rms = np.sqrt(error)
r2 = r2_score(y_test, y_pred)

print("error",error)
print("rms:", rms)
print("r2 score:", r2)

#saving to joblib
joblib.dump(model, 'model.pkl') 
joblib.dump(x_scaler, 'x_scaler.pkl')
joblib.dump(sex_encoder, "sex_encoder.pkl")
joblib.dump(smoker_encoder, "smoker_encoder.pkl")
joblib.dump(region_encoder, "region_encoder.pkl")

print("Model,encoders and scalers saved successfully.")
