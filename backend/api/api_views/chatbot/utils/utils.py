import pandas as pd

from api.models import Disease
from api.serializers import DiseaseSerializer

from api.utils.chatbot import symptom_based


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


def handle_chatbot_response(tag_from_req, msg, model, context, language='en'):
    """
    Handles the chatbot response based on the provided tag and message.

    Args:
        tag_from_req (str): The tag received from the request.
        msg (str): The user's message.
        model: The chatbot model.
        context (dict): The context dictionary to store response and other data.
        language (str, optional): The language of the response. Defaults to 'en'.

    Returns:
        dict: The updated context dictionary.
    """
    if tag_from_req != 'identify_diseases_by_symptoms':
        intent_predictions = model.predict_intent(msg)

        tag = intent_predictions[0]

        response = model.generate_response(tag)
        print(f'INFO: Predicted intent: {tag}\n\t Response: {response}')

        if language == 'si':
            # encode the Sinhala text using UTF-8 encoding
            response = response.encode('utf-8')

        context['response'] = response
        context['tag'] = tag

        # returns response and all diseases in the db for the dropdown
        if tag == 'banana_disease_info' or tag == 'management_strategies':
            diseases = Disease.objects.all()

            serializer = DiseaseSerializer(
                diseases,
                fields=['id', 'name', 'name_display'],
                many=True
            )
            context['diseases'] = serializer.data
    else:
        if language == 'en':
            disease_data = Disease.objects.values('name', 'symptom_description')
            df = pd.DataFrame.from_records(disease_data)
            # ! TODO make columns dynamic
            df = df.rename(columns={'name': 'Disease/Pest', 'symptom_description': 'Description'})

            predicted_diseases = symptom_based.find_top_k_diseases(
                symptoms=msg,
                df=df,
                k=3,
                verbose=True
            )
            print(f'INFO: Predicted Diseases: {predicted_diseases}')

            diseases = []

            for disease in predicted_diseases:
                disease_name = disease[0]
                obj = Disease.objects.get(name=disease_name)
                payload = {
                    'id': obj.id,
                    'name_display': obj.get_name_display(),
                    'confidence': disease[1]
                }
                diseases.append(payload)

            context['diseases'] = diseases
            context['response'] = 'Those are the diseases that fits the description'
    return context
