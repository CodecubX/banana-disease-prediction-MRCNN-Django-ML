import pickle
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
import os
from django.conf import settings

SAVED_DIR = 'api/utils/watering_plan/helpers/'
SOIL_TYPE_MODEL_PATH = 'api/utils/watering_plan/saved_models/soil_type_identification_model.h5'
WATER_PLAN_MODEL_PATH = 'api/utils/watering_plan/saved_models/water_plan_gradientboost_classifier.pkl'

CLASSES = ['Black Soil', 'Cinder Soil', 'Laterite Soil', 'Peat Soil', 'Yellow Soil']


def predict_watering_plan(data, model_file=WATER_PLAN_MODEL_PATH, ordinal_encoder_file=f'{SAVED_DIR}ordinal_encoder.pkl', one_hot_encoder_file=f'{SAVED_DIR}one_hot_encoder.pkl', scaler_file=f"{SAVED_DIR}watering_plan_scaler.pkl", ordinal_encoded_target_file=f'{SAVED_DIR}ordinal_encoder_target_var.pkl'):
    # Ordinal columns (ones that have an order)
    ordinal_columns = ['organic_matter_content', 'slope']
    # Categorical columns (ones without order, already encoded)
    categorical_columns = ['soil_type', 'leaf_color', 'soil_texture',
                           'soil_color', 'water_source', 'irrigation_method',
                           'fertilizer_used_last_season', 'crop_rotation',
                           'pest_disease_infestation']

    # Non-categorical columns
    non_categorical_columns = ['pH', 'soil_moisture', 'avg_temperature', 'avg_rainfall',
                               'plant_height', 'stem_diameter', 'plant_density',
                               'temperature', 'humidity', 'rainfall']

    df = pd.DataFrame.from_dict([data])

    # Load the one-hot encoder
    with open(one_hot_encoder_file, 'rb') as f:
        one_hot_encoder = pickle.load(f)

    # Load the ordinal encoder
    with open(ordinal_encoder_file, 'rb') as f:
        ordinal_encoder = pickle.load(f)

    one_hot_encoded_data = one_hot_encoder.transform(df[categorical_columns])
    ordinal_encoded_data = ordinal_encoder.transform(df[ordinal_columns])

    # Get numerical data columns into a ndarray
    numerical_columns = df[non_categorical_columns].values

    # Create independent variables
    x = np.hstack([
        one_hot_encoded_data,
        ordinal_encoded_data,
        numerical_columns
    ])

    # Load the scaler
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)

    x = scaler.transform(x)

    # Load the model from the file
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Perform inference
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)

    # Reverse transform the predicted labels
    with open(ordinal_encoded_target_file, 'rb') as f:
        ordinal_encoder_target = pickle.load(f)

    # converts [[9.97596309e-01 1.23045918e-05 2.39138657e-03]] to [[0 1 2]]
    class_indexes = np.arange(probabilities.shape[1]).reshape(1, -1)
    # to [[0] [1] [2]]
    formatted_class_indexes = np.transpose(class_indexes)

    class_names = ordinal_encoder_target.inverse_transform(formatted_class_indexes).flatten()

    # Create a dictionary of class names and probabilities
    result = {}
    for class_name, probability in zip(class_names, probabilities[0]):
        result[class_name] = format(probability, '.5f')

    top_prediction = ordinal_encoder_target.inverse_transform(predictions.reshape(1, -1)).flatten()[0]

    return top_prediction, result


def get_processed_input_img(image_path, size=224):
    test_img = cv2.imread(image_path)
    test_img = cv2.resize(test_img, dsize=(size, size), interpolation=cv2.INTER_AREA)

    test_img = test_img.reshape((1, size, size, 3)).astype(np.float32)

    return test_img/225


def predict_soil_type(img_path, model_path=SOIL_TYPE_MODEL_PATH):
    # load trained model
    loaded_model = load_model(model_path)
    processed_img = get_processed_input_img(img_path)
    pred = loaded_model.predict(processed_img)

    # inversely sorted array with indexes
    best_idx = (-pred).argsort()[0]
    # convert them to text
    to_text = [CLASSES[i] for i in best_idx]
    # return top prediction
    return to_text[0]






