from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response

from api.models import Disease, DiseaseQuestionnairePrediction
from api.serializers import DiseaseCureSerializer, DiseaseCureSinhalaSerializer
from api.utils.chatbot.questionnaire_based.predictor import predict_disease
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

    def get_object(self, name):
        return Disease.objects.get(name=name)

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
            disease = self.get_object(name=top_prediction)
            serializer = serializer_class(disease, context={'request': request})
            context['disease'] = serializer.data
        except Disease.DoesNotExist:
            context['disease'] = None
            context['error'] = f'No disease record for {top_prediction} found in database'

        try:
            disease_questionnaire_instance = DiseaseQuestionnairePrediction(
                leaf_color=sample_data.get('Leaf Color'),
                leaf_spots=sample_data.get('Leaf Spots'),
                leaf_wilting=sample_data.get('Leaf Wilting'),
                leaf_curling=sample_data.get('Leaf Curling'),
                stunted_growth=sample_data.get('Stunted Growth'),
                stem_color=sample_data.get('Stem Color'),
                root_rot=sample_data.get('Root Rot'),
                abnormal_fruiting=sample_data.get('Abnormal Fruiting'),
                presence_of_pests=sample_data.get('Presence of Pests'),
                user=request.user,
                disease=self.get_object(name=top_prediction),
                probabilities=predictions
            )
            disease_questionnaire_instance.save()
            print(f'INFO: Saved to history')
        except Exception as e:
            print(f'INFO: Failed to save predictions to database{e}')
            context['error'] = 'Failed to save predictions to database'

        if language == 'si':
            return Response(context, status=status.HTTP_200_OK, content_type='text/plain; charset=utf-8')

        return Response(context, status=status.HTTP_200_OK)
