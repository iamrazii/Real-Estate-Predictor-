from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the complete trained pipeline (preprocessing + model)
model = joblib.load("static/price_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            city_location = request.form['city_location_grouped']
            property_type = request.form['property_type']
            purpose = request.form['purpose']
            bedrooms = int(request.form['bedrooms'])
            baths = int(request.form['baths'])
            area = float(request.form['area'])

            # 2. Feature engineering
            total_rooms = bedrooms + baths
            log_area = np.log1p(area)
            
            # 3. Create input DataFrame
            input_df = pd.DataFrame([{
            'city_location_grouped': city_location,
            'property_type': property_type,
            'purpose': purpose,
            'Total_Rooms': total_rooms,
            'log_area': log_area
                        }])
            
            # 4. Predict using the pipeline
            log_price = model.predict(input_df)[0]
            predicted_price = np.expm1(log_price)
               
            # 5. Show result
            return render_template('predict.html', prediction=f"PKR {round(predicted_price):,}")
         
        except Exception as e:
            return render_template('predict.html', prediction=f"Error: {str(e)}")

    return render_template('predict.html')

# New AJAX endpoint for predictions
@app.route('/predict_ajax', methods=['POST'])
def predict_ajax():
    try:
        city_location = request.form['city_location_grouped']
        property_type = request.form['property_type']
        purpose = request.form['purpose']
        bedrooms = int(request.form['bedrooms'])
        baths = int(request.form['baths'])
        area = float(request.form['area'])

        # Feature engineering
        total_rooms = bedrooms + baths
        log_area = np.log1p(area)
        
        # Create input DataFrame
        input_df = pd.DataFrame([{
            'city_location_grouped': city_location,
            'property_type': property_type,
            'purpose': purpose,
            'Total_Rooms': total_rooms,
            'log_area': log_area
        }])
        
        # Predict using the pipeline
        log_price = model.predict(input_df)[0]
        predicted_price = np.expm1(log_price)
           
        # Return JSON response
        return jsonify({
            'success': True,
            'prediction': f"PKR {round(predicted_price):,}"
        })
     
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Data Analysis Route
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)