import joblib
import pandas as pd

model=  joblib.load('model.pkl')
x_scaler= joblib.load('x_scaler.pkl')
sex_encoder= joblib.load("sex_encoder.pkl")
smoker_encoder= joblib.load("smoker_encoder.pkl")
region_encoder= joblib.load("region_encoder.pkl")

age = int(input("Enter Age: "))
sex = input("Enter Sex (male/female): ").lower()
bmi = float(input("Enter BMI: "))
children = int(input("Enter Number of Children: "))
smoker = input("Smoker? (yes/no): ").lower()
region = input("Enter Region (southwest/southeast/northwest/northeast): ").lower()

sex_encoded = sex_encoder.transform([sex])[0]
smoker_encoded = smoker_encoder.transform([smoker])[0]
region_encoded = region_encoder.transform([region])[0]

sample = pd.DataFrame({
    'age': [age],
    'sex': [sex_encoded],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker_encoded],
    'region': [region_encoded]
})
new_charges_scaled = x_scaler.transform(sample)

prediction = model.predict(new_charges_scaled)

print("prediction of the medical insurance cost:", prediction[0])