
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('wordnet')


def preprocess_text(text, language='english'):
    """
    Preprocesses text by removing punctuation, converting to lowercase, and removing extra white spaces.
    Parameters:
        text (str): The text to preprocess.
        language (str): The language of the text. Valid values are 'sinhala' and 'english'.
    Returns:
        str: The preprocessed text.
    """
    try:
        # Validate the language parameter
        if language not in ['sinhala', 'english']:
            raise ValueError("Invalid language parameter. Must be 'sinhala' or 'english'.")

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        if language == 'english':
            # Convert to lowercase
            text = text.lower()

            # Remove extra whitespaces
            text = re.sub(r'\s+', ' ', text)

        return text

    except Exception as e:
        raise ValueError("Error preprocessing text: " + str(e))


def tokenize_text(text, language='english'):
    """
    Tokenizes text by splitting it into individual words and lemmatizes the tokens.
    Lemmatization only done for english texts
    Parameters:
        text (str): The text to tokenize.
        language (str): The language of the text. Valid values are 'sinhala' and 'english'.
    Returns:
        list: A list of tokens.
    """
    try:
        # Validate the language parameter
        if language not in ['sinhala', 'english']:
            raise ValueError("Invalid language parameter. Must be 'sinhala' or 'english'.")

        # Tokenize the text using NLTK
        tokens = word_tokenize(text)

        if language == 'english':
            # Lemmatize the tokens using NLTK's WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            # lemmatized tokens
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    except Exception as e:
        raise ValueError("Error tokenizing text: " + str(e))


def calculate_tfidf_scores(df, symptom, language='english'):
    """
    Calculates the TF-IDF scores for a list of disease descriptions and a symptom.
    Parameters:
        df (pd.DataFrame): A dataframe of disease/pest, descriptions to compare the symptom against.
        symptom (str): The symptom to calculate the TF-IDF scores for.
        language (str): The language of the text. Valid values are 'sinhala' and 'english'.
    Returns:
        list: A list of tuples, where each tuple contains the disease/pest and its corresponding TF-IDF score.
    """
    try:
        # Validate the language parameter
        if language not in ['sinhala', 'english']:
            raise ValueError("Invalid language parameter. Must be 'sinhala' or 'english'.")

        disease_descriptions = df["Description"].tolist()
        # Preprocess the symptom
        symptom = preprocess_text(symptom, language)

        # Tokenize the symptom
        symptom_tokens = tokenize_text(symptom, language)

        # Preprocess the disease descriptions and tokenize them
        preprocessed_descriptions = [preprocess_text(d) for d in disease_descriptions]
        description_tokens = [tokenize_text(d) for d in preprocessed_descriptions]

        # Combine the symptom and disease description tokens
        all_tokens = description_tokens.copy()
        all_tokens.append(symptom_tokens)

        # Convert the tokens to strings for the TF-IDF vectorizer
        all_strings = [' '.join(tokens) for tokens in all_tokens]

        # Calculate the TF-IDF scores
        if language == 'english':
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        else:
            tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_strings)
        symptom_tfidf = tfidf_matrix[-1]
        description_tfidf = tfidf_matrix[:-1]

        diseases = df['Disease/Pest']

        scores = list(zip(diseases, (symptom_tfidf * description_tfidf.T).A[0]))

        # Sort the scores in descending order
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        return sorted_scores

    except Exception as e:
        raise ValueError("Error calculating TF-IDF scores: " + str(e))