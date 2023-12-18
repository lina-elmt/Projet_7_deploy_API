from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Chargement du modèle Scikit-Learn
model = joblib.load('modele.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    predictions = model.predict_proba(data['features'])[:, 1] 
    
    # Retourner les prédictions au format JSON
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(port = 5000)