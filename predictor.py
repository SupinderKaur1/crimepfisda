import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict_crime_api():
    data = request.json
    print("Request Data:", data)  # Debugging

    # Extract features from the JSON object
    area = data['area']
    age = data['age']
    sex = data['sex']
    day = data['day']

    # Load the model and LabelEncoders
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('day of week_encoder.pkl', 'rb') as file:  # Make sure filename matches exactly
        le_day = pickle.load(file)

    with open('AREA NAME_encoder.pkl', 'rb') as file:  # Make sure filename matches exactly
        le_area = pickle.load(file)

    with open('Vict Sex_encoder.pkl', 'rb') as file:  # Make sure filename matches exactly
        le_sex = pickle.load(file)

    with open('Crm Cd Desc_encoder.pkl', 'rb') as file:  # Make sure filename matches exactly
        le_crime = pickle.load(file)

    # Encode inputs
    sex_encoded = le_sex.transform([sex])[0]
    day_encoded = le_day.transform([day])[0]
    area_encoded = le_area.transform([area])[0]


    # Predict
    prediction = model.predict([[area_encoded, age, sex_encoded, day_encoded]])

    # Decode the predicted crime
    predicted_crime = le_crime.inverse_transform(prediction)[0]

    # Return the prediction as a JSON response
    response = jsonify(predicted_crime=predicted_crime)
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5002)
