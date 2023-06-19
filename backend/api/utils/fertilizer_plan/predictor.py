
import pickle
import numpy as np
import pandas as pd


SAVED_DIR = 'api/utils/fertilizer_plan'
MODEL_DIR = f'{SAVED_DIR}/saved_models'
HELPER_DIR = f'{SAVED_DIR}/helpers'

MODEL_CONFIG = {
    'fertilizer_type': {
        'target': 'fertilizer_type',
        'model_file': f'{MODEL_DIR}/fertilizer_type_classifier.pkl',
        'ordinal_encoder_file': f'{HELPER_DIR}/ordinal_encoder.pkl',
        'one_hot_encoder_file': f'{HELPER_DIR}/one_hot_encoder_fertilizer_type.pkl',
        'scaler_file': f"{HELPER_DIR}/fertilizer_type_scaler.pkl",
        'ordinal_encoded_target_file': f'{HELPER_DIR}/ordinal_encoder_target_var_fertilizer_type.pkl'
    },
    'fertilizer_plan': {
        'target': 'fertilizer_plan',
        'model_file': f'{MODEL_DIR}/fertilizer_plan_classifier.pkl',
        'ordinal_encoder_file': f'{HELPER_DIR}/ordinal_encoder.pkl',
        'one_hot_encoder_file': f'{HELPER_DIR}/one_hot_encoder_fertilizer_plan.pkl',
        'scaler_file': f"{HELPER_DIR}/fertilizer_plan_scaler.pkl",
        'ordinal_encoded_target_file': f'{HELPER_DIR}/ordinal_encoder_target_var_fertilizer_plan.pkl'
    }
}


def predict(data, target='fertilizer_type', model_file='fertilizer_type_classifier.pkl', ordinal_encoder_file='ordinal_encoder.pkl', one_hot_encoder_file='one_hot_encoder_fertilizer_type.pkl', scaler_file="fertilizer_type_scaler.pkl", ordinal_encoded_target_file='ordinal_encoder_target_var_fertilizer_type.pkl'):
    """
    Performs prediction using a trained model on the given data.

    Parameters:
        data (dict): A dictionary containing the input data for prediction.
        target (str): The target variable for prediction. Default is 'fertilizer_type'.
        model_file (str): The file name of the trained model. Default is 'fertilizer_type_classifier.pkl'.
        ordinal_encoder_file (str): The file name of the ordinal encoder. Default is 'ordinal_encoder.pkl'.
        one_hot_encoder_file (str): The file name of the one-hot encoder. Default is 'one_hot_encoder_fertilizer_type.pkl'.
        scaler_file (str): The file name of the scaler. Default is 'fertilizer_type_scaler.pkl'.
        ordinal_encoded_target_file (str): The file name of the ordinal encoded target variable. Default is 'ordinal_encoder_target_var_fertilizer_type.pkl'.

    Returns:
        tuple: A tuple containing the top prediction and a dictionary of class names with corresponding probabilities.
            - top_prediction (str): The top predicted class label.
            - result (dict): A dictionary mapping class names to their corresponding probabilities.

    Note:
        - This function assumes that the required files for the model, encoders, and scaler are stored in a specific directory.
          Make sure to provide the correct file names and paths when calling this function.

    Example usage:
        sample_data = {
            'pH': 6.2,
            'organic_matter_content': 'low',
            ...
        }
        configs = {
            'target': 'fertilizer_plan',
            'model_file': 'fertilizer_plan_classifier.pkl',
            'ordinal_encoder_file': 'ordinal_encoder.pkl',
            'one_hot_encoder_file': 'one_hot_encoder_fertilizer_plan.pkl',
            'scaler_file': "fertilizer_plan_scaler.pkl",
            'ordinal_encoded_target_file': 'ordinal_encoder_target_var_fertilizer_plan.pkl'
        }
        fertilizer_type_pred, confidence = predict(sample_data.-, **configs)
    """

    # Ordinal columns (ones that have an order)
    ordinal_columns = ['organic_matter_content', 'slope']
    # Categorical columns (ones without order, already encoded)
    categorical_columns = ['soil_type', 'leaf_color', 'soil_texture',
                           'soil_color', 'water_source', 'irrigation_method',
                           'fertilizer_used_last_season', 'crop_rotation',
                           'pest_disease_infestation']
    if target == 'fertilizer_plan':
        categorical_columns.append('fertilizer_type')

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

    # conert [[9.97596309e-01 1.23045918e-05 2.39138657e-03]] to [[0 1 2]]
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