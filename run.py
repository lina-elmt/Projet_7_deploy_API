from flask import Flask, request, jsonify
import json
import joblib
import pandas as pd
import shap
shap.initjs()

app = Flask(__name__)

model = joblib.load('modele.pkl')

explainer = shap.TreeExplainer(model)

@app.route('/',methods = ['GET'])
def route_works():
    return "OK"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    data = json.loads(data)

    predictions = model.predict_proba(pd.DataFrame([data]))[:, 1].tolist()
    
    shap_values = explainer.shap_values(pd.DataFrame([data])).tolist()
    
    return jsonify([predictions, shap_values])

if __name__ == '__main__':
    app.run(port = 5000)