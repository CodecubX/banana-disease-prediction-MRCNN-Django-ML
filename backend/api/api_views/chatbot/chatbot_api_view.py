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
        msg, is_identification, language = build_model_data(request.data, request.query_params)

        pred = self.model.get_predictions(msg)

        context = {
            'response': pred,
            'is_identification': is_identification,
            'language': language,
        }
        return Response(context, status=status.HTTP_200_OK)