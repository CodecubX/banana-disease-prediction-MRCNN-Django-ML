from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response

from api.utils import ChatBot
from .utils.utils import build_model_data, handle_chatbot_response


class ChatBotAPIView(APIView):
    """ Handles chatbot related operations"""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        """
        Handles chat bot training
        """
        try:
            language = request.query_params.get('language')
            chatbot = ChatBot(language=language)
            chatbot.train()
        except Exception as e:
            print(f'ERROR: {e}')
            return Response({'error': 'Something went wrong while training'}, status=status.HTTP_200_OK)
        return Response({'detail': 'Model training completed'}, status=status.HTTP_200_OK)

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

        try:
            model = ChatBot(language=language)
            model.load_model_and_data()

        except Exception as e:
            print(f'ERROR: {e}')
            return Response({'error': 'Something went wrong with the chatbot'}, status=status.HTTP_409_CONFLICT)

        context = {
            'response': '',
            'language': language,
            'tag': tag_from_req
        }

        try:
            context = handle_chatbot_response(tag_from_req, msg, model, context, language=language)
        except IndexError:
            print(f'ERROR: Responses empty in intents')
            return Response({'error': 'Chat bot responses not found'}, status=status.HTTP_409_CONFLICT)
        if language == 'si':
            return Response(context, status=status.HTTP_200_OK, content_type='text/plain; charset=utf-8')
        return Response(context, status=status.HTTP_200_OK)

