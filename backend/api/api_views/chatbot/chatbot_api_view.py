from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response


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
        # ! TODO add validations and separate method
        # data from url query params
        language = request.query_params.get('language')
        # post data
        msg = request.data.get('msg')
        is_identification = request.data.get('is_identification')

        context = {
            'response': msg,
            'is_identification': is_identification,
            'language': language,
        }
        return Response(context, status=status.HTTP_200_OK)