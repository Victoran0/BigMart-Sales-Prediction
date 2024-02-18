from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("home.html")


@app.route('/predict', methods=['POST', 'GET'])
def result():
    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = float(
        request.form['outlet_establishment_year'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])

# we convert our inputs into numpy 2d array
    X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                 outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])
    scaler_path = os.path.join(
        r'C:\Users\User\Data Science\BigMart Sales Prediction\models\sc.sav')
    sc = joblib.load(scaler_path)
    X_std = sc.transform(X)

    model_path = os.path.join(
        r'C:\Users\User\Data Science\BigMart Sales Prediction\models\lr.sav')

    model = joblib.load(model_path)

    y_pred = model.predict(X_std)

    return jsonify({"Prediction": float(y_pred)})


if __name__ == '__main__':
    app.run(debug=True, port=9457)
