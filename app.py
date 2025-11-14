from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved ML model
model = joblib.load('ipl_score_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    batting_team = request.form['bat_team']
    bowling_team = request.form['bowl_team']
    overs = float(request.form['over'])
    runs = float(request.form['score'])
    wickets = int(request.form['wickets'])
    runs_last_5 = float(request.form['runs_last_5'])
    wickets_last_5 = int(request.form['wickets_last_5'])

    # Prepare input
    input_data = pd.DataFrame({
        'bat_team': [batting_team],
        'bowl_team': [bowling_team],
        'overs': [overs],
        'runs': [runs],
        'wickets': [wickets],
        'runs_last_5': [runs_last_5],
        'wickets_last_5': [wickets_last_5]
    })

    # Make prediction
    prediction = int(model.predict(input_data)[0])

    return render_template('index.html', prediction_text=f"🏏 Predicted Final Score: {prediction} runs")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

