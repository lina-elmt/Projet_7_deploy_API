from flask import Flask, request, jsonify
import json
import joblib
import pandas as pd
import shap
shap.initjs()

app = Flask(__name__)

model = joblib.load('modele.pkl')

explainer = shap.TreeExplainer(model)
sample = pd.read_parquet('sample.parquet')

X = sample.drop(columns = 'TARGET')

shap_values = explainer.shap_values(X)

@app.route('/',methods = ['GET'])
def get_shap_values():
    return jsonify(shap_values.tolist())

@app.route('/distribution', methods=['POST'])
def distribution():
    
    data = request.data.decode('latin1')
    
    X_distribution_0 = sample[sample['TARGET']==0][data].tolist()
    
    X_distribution_1 = sample[sample['TARGET']==1][data].tolist()
    
    return jsonify([X_distribution_0, X_distribution_1])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    data = json.loads(data)

    predictions = model.predict_proba(pd.DataFrame([data]))[:, 1].tolist()
    
    shap_values = explainer(pd.DataFrame([data]))
    
    shap_values_dict = {
    "values": shap_values.values.tolist(),
    "base_values": shap_values.base_values.tolist(),
    "data": shap_values.data.tolist(),
    "feature_names": shap_values.feature_names
    }
    
    shap_values_json = json.dumps(shap_values_dict)
    
    return jsonify([predictions, shap_values_json])

if __name__ == '__main__':
    app.run(port = 5000)