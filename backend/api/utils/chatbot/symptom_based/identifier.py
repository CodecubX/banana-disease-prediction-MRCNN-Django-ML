from .utils.utils import calculate_tfidf_scores


def find_top_k_diseases(symptoms, df, k=None, language='english', verbose=True):
    """
    Finds the top k diseases/pests from the given dataset that match the given symptoms based on their TF-IDF scores.

    Parameters:
        symptoms (str): A string containing the symptoms.
        df (pd.DataFrame): A pandas DataFrame containing the dataset of diseases/pests and their descriptions.
        k (int): An integer specifying the number of top diseases/pests to return. If None, returns all diseases/pests.
        verbose (bool): A boolean indicating whether to print the scores of all diseases/pests or not.

    Returns:
        A list of tuples, where each tuple contains the name of a disease/pest and its corresponding TF-IDF score.
    """

    # Validate the language parameter
    if language not in ['sinhala', 'english']:
        raise ValueError("Invalid language parameter. Must be 'sinhala' or 'english'.")

    # Calculate the TF-IDF scores
    scores = calculate_tfidf_scores(df, symptoms, language)

    if verbose:
        # Print all diseases/pests with the scores
        for disease, score in scores:
            print(disease, score)

    return scores[:k]