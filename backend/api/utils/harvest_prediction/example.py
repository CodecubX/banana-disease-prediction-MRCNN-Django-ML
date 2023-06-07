from .predictor import get_harvest_prediction


sample_data = {
    "Variety": "Amban Banana",
    "Agro-climatic region": "Wet Zone",
    "Plant density(Min=1,Max=5)": 1,
    "Spacing between plants (m)": 2.46,
    'Pesticides used(Yes, No)': 'Yes',
    "Plant generation": str(1),
    "Fertilizer type": "Non Organic",
    "Soil pH": 7.7,
    "Amount of sunlight received": "Low",
    "Watering schedule": "randomly",
    "Number of leaves": 16,
    "Height (m)": 2.0
}

get_harvest_prediction(data=sample_data, verbose=True)