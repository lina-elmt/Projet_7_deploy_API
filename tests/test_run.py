from run import app
import json

def test_route():
    
    response = app.test_client().get('/')

    assert b'OK' == response.data 

def test_predict_api():
    
    data = {
        
       'EXT_SOURCE_2' : 0.44600,
       'DAYS_EMPLOYED' : -2603,
       'AMT_GOODS_PRICE' : 1035000.0,
       'number_of_previous_Active_credits' : 9.0,
       'number_of_previous_Closed_credits' : 4.0,
       'NAME_EDUCATION_TYPE_Higher education' : False,
       'number_of_previous_Refused_credits' : 0.0,
       'percent_month_installments_late_moyen' : 0.000000,
       'AMT_CREDIT' : 1035000.0,
       'AMT_ANNUITY': 27432.0
       
        } 
    
    json_data = json.dumps(data)
    
    response = app.test_client().post('/predict', json = json_data)
    
    assert response.status_code == 200
    
    json_response_prediction = response.get_json()[0]
    
    assert isinstance(json_response_prediction[0], float)
    
    assert 0 <= json_response_prediction[0] <= 1
    
    json_response_shap = response.get_json()[1]
    
    assert len(json_response_shap[0]) == 10
    
    