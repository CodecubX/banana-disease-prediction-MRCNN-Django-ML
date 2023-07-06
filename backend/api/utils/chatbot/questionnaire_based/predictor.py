import pandas as pd
import numpy as np
import pickle

BASE_DIR = 'api/utils/chatbot/'
MODEL_DIR = f'{BASE_DIR}saved_models/'
HELPER_DIR = f'{BASE_DIR}helper/'

# Load the trained model
with open(f'{MODEL_DIR}disease_questionnaire_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the encoders
with open(f'{HELPER_DIR}disease_questionnaire_x_encoder.pkl', 'rb') as file:
    ohe_X = pickle.load(file)

with open(f'{HELPER_DIR}disease_questionnaire_y_encoder.pkl', 'rb') as file:
    ordinal_y = pickle.load(file)


def predict_disease(symptoms):
    """
    predicts the disease based on the given symptoms.

    args:
        symptoms (dict): dictionary containing the symptoms as keys and their corresponding values.

    returns:
        tuple: a tuple containing the predicted disease label and a dictionary of probabilities for the top three predictions.
    """

    # convert user input into a dataframe
    input_data = pd.DataFrame.from_dict([symptoms])

    # perform one-hot encoding
    encoded_data = ohe_X.transform(input_data)

    # make predictions
    pred_proba = model.predict_proba(encoded_data)
    pred = model.predict(encoded_data)

    # convert predictions back to disease labels
    pred = ordinal_y.inverse_transform(pred.reshape(-1, 1)).flatten()
    labels = ordinal_y.inverse_transform(model.classes_.reshape(-1, 1)).flatten()

    # create predictions and probabilities dictionary for the top three predictions
    top_three_pred_proba = {label: format(probability, '.5f') for label, probability in zip(labels[:3], pred_proba[0, np.argsort(pred_proba[0])[::-1]][:3])}

    return pred[0], top_three_pred_proba