import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = 'api/utils/harvest_prediction/'

CLASSIFIER_PATH = f'{BASE_DIR}saved_models/harvest_classifier.pkl'
ORDINAL_ENCODER_PATH = f'{BASE_DIR}helpers/harvest_data_ordinal_encoded.pkl'
ONE_HOT_ENCODER_PATH = f'{BASE_DIR}helpers/harvest_data_one_hot_encoded.pkl'
Y_ENCODER_PATH = f'{BASE_DIR}helpers/y_data_ordinal_encoded.pkl'
SCALER_PATH = f'{BASE_DIR}helpers/harvest_data_scaler.pkl'


def get_harvest_prediction(data, classifier=CLASSIFIER_PATH,
                           ordinal_encoder_obj=ORDINAL_ENCODER_PATH,
                           onehot_encoder_obj=ONE_HOT_ENCODER_PATH, scaler_obj=SCALER_PATH,
                           y_encoder_obj=Y_ENCODER_PATH, top_k=3, verbose=False):
    """
    This function predicts the harvest of a banana crop based on the given input data.
    It loads the pre-trained model, encoders, and scaler using pickle and applies them on the input data to get the predictions.

    Args:
        data (dict): A dictionary containing the input data for the prediction. It must include the following keys:
            - 'Variety': The variety of the banana crop. (str)
            - 'Agro-climatic region': The agro-climatic region where the banana crop is grown. (str)
            - 'Plant density(Min=1,Max=5)': The plant density of the banana crop. (int)
            - 'Spacing between plants (m)': The spacing between plants in meters. (float)
            - 'Plant generation': The generation of the banana crop. (int)
            - 'Fertilizer type': The type of fertilizer used for the banana crop. (str)
            - 'Soil pH': The pH value of the soil. (float)
            - 'Amount of sunlight received': The amount of sunlight received by the banana crop. (str)
            - 'Watering schedule': The watering schedule for the banana crop. (str)
            - 'Number of leaves': The number of leaves on the banana crop. (int)
            - 'Height (m)': The height of the banana crop in meters. (float)
        classifier (str): The file path of the trained classifier object. Defaults to 'harvest_classifier.pkl'.
        ordinal_encoder_obj (str): The file path of the ordinal encoder object for 'Amount of sunlight received' and 'Watering schedule' columns. Defaults to 'harvest_data_ordinal_encoded.pkl'.
        onehot_encoder_obj (str): The file path of the one-hot encoder object for 'Variety', 'Agro-climatic region', and 'Fertilizer type' columns. Defaults to 'harvest_data_one_hot_encoded.pkl'.
        scaler_obj (str): The file path of the scaler object for scaling the encoded data. Defaults to 'harvest_data_scaler.pkl'.
        y_encoder_obj (str): The file path of the ordinal encoder object for the target variable 'Harvest'. Defaults to 'y_data_ordinal_encoded.pkl'.
        top_k (int): Means return top k predictions that has more probabilities
        verbose (bool): Whether to print the predicted probabilities with encoded values. Defaults to False.

    Returns:
        tuple: A tuple containing two elements:
            - The predicted harvest in kilograms (str).
            - A dictionary containing the predicted probabilities for each harvest range. The keys are the harvest ranges and the values are the probabilities.

    Raises:
        KeyError: If the input data dictionary is missing one or more required keys.
        TypeError: If one or more input values have an invalid data type.
        ValueError: If the input values have an invalid value range.

    """

    # Validate input data
    required_keys = [
        'Variety', 'Agro-climatic region', 'Plant density(Min=1,Max=5)', 'Spacing between plants (m)',
        'Plant generation',
        'Fertilizer type', 'Soil pH', 'Amount of sunlight received', 'Watering schedule',
        'Number of leaves', 'Height (m)'
    ]
    for key in required_keys:
        if (key not in data) or data[key] == None:
            raise KeyError(f"Input data is missing '{key}' key.")

    # Validate values for each key in data
    valid_values = {
        'Variety': ['Amban Banana', 'Mysore Banana', 'Pisang Awak Banana', 'Silk Banana', 'Anamalu Banana'],
        'Agro-climatic region': ['Wet Zone', 'Dry Zone', 'Intermediate Zone'],
        'Plant density(Min=1,Max=5)': [1, 2, 3, 4, 5],
        'Pesticides used(Yes, No)': ['Yes', 'No'],
        'Fertilizer type': ['Non Organic', 'None', 'Organic', 'Both used'],
        'Soil pH': (1, 14),
        'Amount of sunlight received': ['Low', 'Moderate', 'High'],
        'Watering schedule': ['none', 'randomly', 'twice a week', '3 times a week', 'Daily'],
        'Plant generation': ['1', '2', '4', '3', '5', 'more than 5']
    }

    # Validate values for each key in data
    for key in valid_values:
        if key not in data:
            raise KeyError(f"Input data is missing '{key}' key.")

        if key in valid_values:
            value = data[key]
            if isinstance(value, str):
                if value not in valid_values[key]:
                    raise ValueError(f"Invalid value '{value}' for key '{key}'. Valid values are {valid_values[key]}.")
            elif isinstance(value, (int, float)):
                if isinstance(valid_values[key], tuple):
                    if not valid_values[key][0] <= value <= valid_values[key][1]:
                        raise ValueError(
                            f"Invalid value '{value}' for key '{key}'. Valid values are between {valid_values[key][0]} and {valid_values[key][1]}.")
                elif isinstance(valid_values[key], list):
                    if value not in valid_values[key]:
                        raise ValueError(
                            f"Invalid value '{value}' for key '{key}'. Valid values are {valid_values[key]}.")
            else:
                raise TypeError(f"Invalid data type for key '{key}'.")
        else:
            raise KeyError(f"Invalid key '{key}' in input data.")

    # validate and Load the model from file
    try:
        with open(classifier, 'rb') as f:
            harvest_classifier = pickle.load(f)
    except FileNotFoundError:
        raise Exception(f'The trained model object `{classifier}` is not found.')

    # Load the encoder objects using pickle
    try:
        with open(ordinal_encoder_obj, 'rb') as f:
            oe = pickle.load(f)
    except FileNotFoundError:
        raise Exception(f'The ordinal encoder objects `{ordinal_encoder_obj}`are not found.')
    try:
        with open(onehot_encoder_obj, 'rb') as f:
            ohe = pickle.load(f)
    except FileNotFoundError:
        raise Exception(f'The onehot encoder objects `{onehot_encoder_obj}` are not found.')

    # Ordinal Encode categorical columns
    oe_cols = ['Amount of sunlight received', 'Watering schedule', 'Plant generation']
    # One-hot Encode categorical columns
    ohe_cols = ['Variety', 'Agro-climatic region', 'Fertilizer type', 'Pesticides used(Yes, No)']
    # columns that have numerical values
    other_columns = [
        'Plant density(Min=1,Max=5)',
        'Spacing between plants (m)',
        'Soil pH',
        'Number of leaves',
        'Height (m)'
    ]

    # data to dataframe
    data = pd.DataFrame.from_dict([data])

    # Ordinal encode data
    try:
        oe_data = oe.transform(data[oe_cols])
    except ValueError:
        raise ValueError('The values of "Amount of sunlight received" and/or "Watering schedule" are not valid.')

    # One-hot encode data
    try:
        ohe_data = ohe.transform(data[ohe_cols])
    except ValueError:
        raise ValueError('The values of "Variety", "Agro-climatic region", and/or "Fertilizer type" are not valid.')

    # get other column data
    non_categorical_data = data[other_columns].values
    # combine encoded data
    encoded_data = np.hstack([ohe_data, oe_data, non_categorical_data])

    # Load the scaler objects using pickle
    try:
        with open(scaler_obj, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        raise Exception(f'The scaler object `{scaler_obj}` is not found')

    try:
        normalized_data = scaler.transform(encoded_data)
    except ValueError as e:
        print(f"Error occurred while normalizing data: {e}")

    # get predictions
    pred_proba = harvest_classifier.predict_proba(normalized_data)
    pred = harvest_classifier.predict(normalized_data)

    # Load the encoder objects using pickle
    try:
        with open(y_encoder_obj, 'rb') as f:
            y_oe = pickle.load(f)
    except FileNotFoundError:
        raise Exception(f'The target variable ordinal encoder {y_encoder_obj} oject is not found')

    pred = y_oe.inverse_transform(pred.reshape(-1, 1)).flatten()
    labels = y_oe.inverse_transform(harvest_classifier.classes_.reshape(-1, 1)).flatten()

    # create predictions and probabilities dict
    pred_proba = {label: format(probability, '.5f') for label, probability in zip(labels, pred_proba[0])}

    # Sort the dictionary based on probabilities in descending order
    sorted_proba = sorted(pred_proba.items(), key=lambda x: float(x[1]), reverse=True)

    # Get the top k probabilities
    top_k_proba = sorted_proba[:top_k]
    # Create a new dictionary with the top k probabilities
    top_k_proba = {label: probability for label, probability in top_k_proba}

    if verbose:
        # Print the predicted probabilities with encoded values
        print('Probabilities for each class: ')
        for i in pred_proba:
            print(f'\t{i}: {pred_proba[i]}')

    return pred[0], top_k_proba


def calculate_harvesting_time(average_harvesting_time, date):
    """
    Calculates the estimated harvesting time for a banana plant based on its average harvesting time and age in days.

    Args:
        average_harvesting_time (int): The average harvesting time for the banana plant in days.
        date (str): The age of the banana plant in days.

    Returns:
        str: Harvest date
    """
    date = datetime.strptime(date, '%Y-%m-%d')
    # Calculate the harvesting time by adding the average_harvesting_time days
    date = date + timedelta(days=average_harvesting_time)
    estimated_harvesting_time = date.strftime('%Y-%m-%d')

    return f"The estimated harvesting is on {estimated_harvesting_time}"
