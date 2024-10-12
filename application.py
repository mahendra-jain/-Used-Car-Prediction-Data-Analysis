from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car_data = pd.read_csv('quikr_car_cleaned.csv')


@app.route('/')
def index():
    companies = sorted(car_data['company'].unique())
    car_models = sorted(car_data['name'].unique())
    year = sorted(car_data['year'].unique(), reverse=True)
    fuel_type = sorted(car_data['fuel_type'].unique())
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        kms_driven = int(request.form.get('kms_driven'))

        print("Received data:", company, car_model, year, fuel_type, kms_driven)

        # Prepare data for prediction
        data = pd.DataFrame([[car_model, company, fuel_type]], columns=['name', 'company', 'fuel_type'])
        print("Data:", data)

        # Apply the same encoding used during training
        data_encoded = model.transform(data[['name', 'company', 'fuel_type']])
        print("Encoded Data:", data_encoded)

        # Add numerical columns
        data_numerical = data[['year', 'kms_driven']].values
        data_combined = np.concatenate([data_encoded, data_numerical], axis=1)

        print("Combined Data:", data_combined)

        # Make prediction
        prediction = model.predict(data_combined)
        print("Prediction:", prediction)

        return str(prediction[0])

    except Exception as e:
        print("An error occurred during prediction:", e)
        return "An error occurred during prediction: " + str(e)


if __name__ == "__main__":
    app.run(debug=True)
