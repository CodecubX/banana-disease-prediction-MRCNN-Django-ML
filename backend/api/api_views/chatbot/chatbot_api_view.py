import pandas as pd

from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response

from api.models import Disease

from api.serializers import DiseaseSerializer

from api.utils import ChatBot
from api.utils.chatbot import symptom_based
from .utils.utils import build_model_data


class ChatBotAPIView(APIView):
    """ Handles chatbot related operations"""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        """
        Handles the POST request for chatbot predictions.

        Parameters:
            request (HttpRequest): The HTTP request object.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Response: The HTTP response containing the prediction results and other data.
        """
        msg, tag_from_req, language = build_model_data(request.data, request.query_params)

        model = ChatBot(language=language, mode='development')

        context = {
            'response': '',
            'language': language,
            'tag': tag_from_req
        }

        if tag_from_req != 'identify_diseases_by_symptoms':
            intent_predictions = model.get_predictions(msg)

            tag = intent_predictions[0]['intent']
            response = model.get_response(tag)

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
                # ! TODO improve following
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

        return Response(context, status=status.HTTP_200_OK)
