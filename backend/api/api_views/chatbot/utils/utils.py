def build_model_data(data, query_params):
    """
    Builds the data dictionary for the model using the provided data.

    Parameters:
        data (dict): A dictionary containing the request data.

    Returns:
        dict: The sample data dictionary for the model.
    """
    msg = data.get('msg')
    is_identification = data.get('is_identification')
    language = query_params.get('language')



    return msg, is_identification, language
