import json

from .predictor import predict


sample_data = {
    'pH': 6.2,
    'organic_matter_content': 'low',
    'soil_type': 'Cinder Soil',
    'soil_moisture': '25',
    'avg_temperature': 28,
    'avg_rainfall': 100,
    'plant_height': 50,
    'leaf_color': 'Green',
    'stem_diameter': 3,
    'plant_density': 3,
    'soil_texture': 'Clayey',
    'soil_color': 'Red',
    'temperature': 30,
    'humidity': 20,
    'rainfall': 100,
    'water_source': 'River',
    'irrigation_method': 'Drip',
    'fertilizer_used_last_season': 'organic',
    'crop_rotation': 'yes',
    'pest_disease_infestation': 'no',
    'slope': 'low'
}

model_configs = {
    'fertilizer_type': {
        'target': 'fertilizer_type',
        'model_file': 'fertilizer_type_classifier.pkl',
        'ordinal_encoder_file': 'ordinal_encoder.pkl',
        'one_hot_encoder_file': 'one_hot_encoder_fertilizer_type.pkl',
        'scaler_file': "fertilizer_type_scaler.pkl",
        'ordinal_encoded_target_file': 'ordinal_encoder_target_var_fertilizer_type.pkl'
    },
    'fertilizer_plan': {
        'target': 'fertilizer_plan',
        'model_file': 'fertilizer_plan_classifier.pkl',
        'ordinal_encoder_file': 'ordinal_encoder.pkl',
        'one_hot_encoder_file': 'one_hot_encoder_fertilizer_plan.pkl',
        'scaler_file': "fertilizer_plan_scaler.pkl",
        'ordinal_encoded_target_file': 'ordinal_encoder_target_var_fertilizer_plan.pkl'
    }
}

# predict fertilizer type first then feed that output to fertilizer plan model
fertilizer_type_pred, confidence = predict(sample_data)
print(fertilizer_type_pred)
print(json.dumps(confidence, indent=4))

sample_data['fertilizer_type'] = fertilizer_type_pred

fertilizer_plan_pred, confidence = predict(sample_data, **model_configs['fertilizer_plan'])
print(fertilizer_plan_pred)
print(json.dumps(confidence, indent=4))
