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

predict(sample_data)