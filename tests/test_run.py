from run import app
import json
import shap
import numpy as np

def test_shap():
    
    response = app.test_client().get('/')
    
    assert response.status_code == 200

    assert len(response.data) >1
    
def test_distribution():
    
    response = app.test_client().post('/distribution', data = "Crédits clos".encode("latin1"))
    
    assert response.status_code == 200

    assert len(response.get_json()[0]) >1
    
    assert len(response.get_json()[1]) >1

def test_predict_api():
    
    data = {
        
       'Notation bancaire' : 0.44600,
       'Âge' : 36.00,
       'Crédits en cours' : 9.0,
       'Prix biens consommation' : 1035000.0,
       'Crédits clos' : 4.0,
       'Enseignement supérieur' : False,
       'Crédits refusés' : 0.0,
       'Mois avec retard de paiement' : 0.000000,
       'Montant total prêt' : 1035000.0       
        } 
    
    json_data = json.dumps(data)
    
    response = app.test_client().post('/predict', json = json_data)
    
    assert response.status_code == 200
    
    json_response_prediction = response.get_json()[0]
    
    assert isinstance(json_response_prediction[0], float)
    
    assert 0 <= json_response_prediction[0] <= 1
    
    json_response_shap = response.get_json()[1]
    
    shap_values_dict_reconstructed = json.loads(json_response_shap)

    shap_values_reconstructed = shap.Explanation(
        values=np.array(shap_values_dict_reconstructed['values']),
        base_values=np.array(shap_values_dict_reconstructed['base_values']),
        data=np.array(shap_values_dict_reconstructed['data']),
        feature_names=shap_values_dict_reconstructed['feature_names']
    )[0]
    
    assert len(shap_values_reconstructed) == 9
    
    