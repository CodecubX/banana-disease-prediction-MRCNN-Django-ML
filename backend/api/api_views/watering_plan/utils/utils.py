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
        "pH": data.get('pH'),
        "organic_matter_content": data.get('organic_matter_content'),
        "soil_type": data.get('soil_type'),
        "soil_moisture": data.get('soil_moisture'),
        "avg_temperature": data.get('avg_temperature'),
        "avg_rainfall": data.get('avg_rainfall'),
        "plant_height": data.get('plant_height'),
        "leaf_color": data.get('leaf_color'),
        "stem_diameter": data.get('stem_diameter'),
        "plant_density": data.get('plant_density'),
        "soil_texture": data.get('soil_texture'),
        "soil_color": data.get('soil_color'),
        "temperature": data.get('temperature'),
        "humidity": data.get('humidity'),
        "rainfall": data.get('rainfall'),
        "water_source": data.get('water_source'),
        "irrigation_method": data.get('irrigation_method'),
        "fertilizer_used_last_season": data.get('fertilizer_used_last_season'),
        "crop_rotation": data.get('crop_rotation'),
        "pest_disease_infestation": data.get('pest_disease_infestation'),
        "slope": data.get('slope')
    }
    return sample_data
