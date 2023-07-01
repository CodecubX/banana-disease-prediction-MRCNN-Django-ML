from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response

from api.models import Disease
from api.serializers import DiseaseCureSerializer, DiseaseCureSinhalaSerializer
from api.utils.chatbot.questionnaore_based.predictor import predict_disease
from .utils.utils import build_questionnaire_model_data


class DiseaseQuestionnaireAPIView(APIView):
    """ Handles Disease Questionnaire related operations """

    permission_classes = [permissions.IsAuthenticated]

    def get_serializer_class(self):
        language = self.request.query_params.get('language', 'en')
        if language == 'en':
            return DiseaseCureSerializer
        elif language == 'si':
            return DiseaseCureSinhalaSerializer
        else:
            return DiseaseCureSerializer

    def post(self, request, *args, **kwargs):
        serializer_class = self.get_serializer_class()

        data = request.data
        language = request.query_params.get('language', 'en')
        sample_data = build_questionnaire_model_data(data)

        try:
            top_prediction, predictions = predict_disease(sample_data)
        except Exception as e:
            print(f'ERROR: {e}')
            return Response({'error': 'Something went wrong while running predictor'}, status=status.HTTP_409_CONFLICT)

        context = {
            'top_prediction': top_prediction,
            'probabilities': predictions
        }

        try:
            disease = Disease.objects.get(name=top_prediction)
            serializer = serializer_class(disease)
            context['disease'] = serializer.data
        except Disease.DoesNotExist:
            context['disease'] = None
            context['error'] = f'No disease record for {top_prediction} found in database'

        if language == 'si':
            return Response(context, status=status.HTTP_200_OK, content_type='text/plain; charset=utf-8')

        return Response(context, status=status.HTTP_200_OK)
