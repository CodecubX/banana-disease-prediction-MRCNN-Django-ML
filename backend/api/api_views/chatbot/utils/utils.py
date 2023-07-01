def build_model_data(data, query_params):
    """
    Builds the data dictionary for the model using the provided data.

    Parameters:
        data (dict): A dictionary containing the request data.

    Returns:
        dict: The sample data dictionary for the model.
    """
    msg = data.get('msg')
    tag = data.get('tag', None)
    language = query_params.get('language')

    return msg, tag,  language


def build_questionnaire_model_data(data):
    """
    Builds the data dictionary for the questionnaire model using the provided data.

    Parameters:
        data (dict): A dictionary containing the request data.

    Returns:
        dict: The sample data dictionary for the model.
    """

    # Extract the data from the request and create the data dictionary
    sample_data = {
        'Leaf Color': data.get('leaf_color'),
        'Leaf Spots': data.get('leaf_spots'),
        'Leaf Wilting': data.get('leaf_wilting'),
        'Leaf Curling': data.get('leaf_curling'),
        'Stunted Growth': data.get('stunned_growth'),
        'Stem Color': data.get('stem_color'),
        'Root Rot': data.get('root_rot'),
        'Abnormal Fruiting': data.get('abnormal_fruiting'),
        'Presence of Pests': data.get('presence_of_pests')
    }
    return sample_data
