def build_model_data(data):
    """
    Builds the data dictionary for the model using the provided data.

    Parameters:
        data (dict): A dictionary containing the request data.

    Returns:
        dict: The sample data dictionary for the model.
    """

    # Extract the data from the request
    variety = data.get('variety')
    agro_climatic_region = data.get('agro_climatic_region')
    plant_density = data.get('plant_density')
    spacing_between_plants = data.get('spacing_between_plants')
    pesticides_used = data.get('pesticides_used')
    plant_generation = data.get('plant_generation')
    fertilizer_type = data.get('fertilizer_type')
    soil_ph = data.get('soil_ph')
    amount_of_sunlight = data.get('amount_of_sunlight')
    watering_schedule = data.get('watering_schedule')
    number_of_leaves = data.get('number_of_leaves')
    height = data.get('height')

    # Create the data dictionary
    sample_data = {
        "Variety": variety,
        "Agro-climatic region": agro_climatic_region,
        "Plant density(Min=1,Max=5)": plant_density,
        "Spacing between plants (m)": spacing_between_plants,
        'Pesticides used(Yes, No)': pesticides_used,
        "Plant generation": plant_generation,
        "Fertilizer type": fertilizer_type,
        "Soil pH": soil_ph,
        "Amount of sunlight received": amount_of_sunlight,
        "Watering schedule": watering_schedule,
        "Number of leaves": number_of_leaves,
        "Height (m)": height
    }
    return sample_data
