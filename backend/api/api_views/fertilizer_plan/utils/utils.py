def build_model_data(data):
    """
    Builds the data dictionary for the model using the provided data.

    Parameters:
        data (dict): A dictionary containing the request data.

    Returns:
        dict: The sample data dictionary for the model.
    """

    # Create the data dictionary
    sample_data = {
        "pH": float(data.get('pH')),
        "organic_matter_content": data.get('organic_matter_content'),
        "soil_type": data.get('soil_type'),
        "soil_moisture": float(data.get('soil_moisture')),
        "avg_temperature": float(data.get('avg_temperature')),
        "avg_rainfall": float(data.get('avg_rainfall')),
        "plant_height": float(data.get('plant_height')),
        "leaf_color": data.get('leaf_color'),
        "stem_diameter": float(data.get('stem_diameter')),
        "plant_density": int(data.get('plant_density')),
        "soil_texture": data.get('soil_texture'),
        "soil_color": data.get('soil_color'),
        "temperature": float(data.get('temperature')),
        "humidity": float(data.get('humidity')),
        "rainfall": float(data.get('rainfall')),
        "water_source": data.get('water_source'),
        "irrigation_method": data.get('irrigation_method'),
        "fertilizer_used_last_season": data.get('fertilizer_used_last_season'),
        "crop_rotation": data.get('crop_rotation'),
        "pest_disease_infestation": data.get('pest_disease_infestation'),
        "slope": data.get('slope')
    }
    return sample_data


def get_top_predictions(probabilities, n=3):
    """
    Get the top N predictions from a dictionary of probabilities.

    Parameters:
        probabilities (dict): A dictionary mapping predictions to their corresponding probabilities.
        n (int): The number of top predictions to retrieve. Default is 3.

    Returns:
        list: A list of dictionaries representing the top predictions, sorted by probability in descending order.
            Each dictionary has 'plan' and 'probability' as keys.

    Example usage:
        top_predictions = get_top_predictions(top_probabilities, n=3)
    """
    sorted_probabilities = sorted(probabilities.items(), key=lambda x: float(x[1]), reverse=True)[:n]
    formatted_predictions = [{"dose": key, "probability": value} for key, value in sorted_probabilities]
    return formatted_predictions
