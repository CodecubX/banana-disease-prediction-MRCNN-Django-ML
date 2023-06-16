def filter_and_calculate_area(predictions):
    """
    Filters sub-objects with an area of 0(removed) and calculates average confidence scores and total area for each class.

    Args:
        predictions (dict): A dictionary containing class names as keys and a list of sub-objects as values.

    Returns:
        list: A list containing the filtered and sorted results with class names as keys, and average
              confidence scores and total area as values.

    """
    # Remove sub-objects with an area of 0
    filtered_data = {}

    for class_name, objects in predictions.items():
        filtered_objects = [obj for obj in objects if obj['area'] != 0]
        if filtered_objects:
            filtered_data[class_name] = filtered_objects

    # Calculate average confidence scores and total area for each class
    sorted_arr = []
    for class_name, objects in filtered_data.items():
        avg_confidence = float(sum(obj['score'] for obj in objects) / len(objects))
        total_area = int(sum(obj['area'] for obj in objects))
        disease_dict = {
            'disease_name': class_name,
            'avg_confidence': format(avg_confidence, '.3f'),
            'total_area': total_area
        }
        sorted_arr.append(disease_dict)
    # Sort the result dictionary based on total area in descending order
    sorted_arr = sorted(sorted_arr, key=lambda x: x['total_area'], reverse=True)

    return sorted_arr
