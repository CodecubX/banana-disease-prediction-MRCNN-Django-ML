from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response

from api.utils import ChatBot
from .utils.utils import build_model_data


class ChatBotAPIView(APIView):
    """ Handles chatbot related operations"""

    permission_classes = [permissions.IsAuthenticated]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = ChatBot()  # Initialize the ChatBot instance specific to each ChatBotAPIView instance

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
        msg, language = build_model_data(request.data, request.query_params)

        intent_predictions = self.model.get_predictions(msg)

        tag = intent_predictions[0]['intent']

        if tag == 'banana_disease_info':
            pass
        elif tag == 'management_strategies':
            pass
        elif tag == 'identify_diseases_by_symptoms':
            pass
        else:
            response = self.model.get_response(tag)

        context = {
            'response': response,
            'language': language,
            'tag': tag
        }
        return Response(context, status=status.HTTP_200_OK)