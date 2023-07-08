from rest_framework import permissions
from rest_framework.generics import ListAPIView
from rest_framework.response import Response

from api.models import DiseaseQuestionnairePrediction

from api.serializers import DiseaseQuestionnairePredictionSerializer, DiseaseQuestionnairePredictionSinhalaSerializer


class DiseaseQuestionnairePredictionHistoryAPIView(ListAPIView):
    """ Handles Disease Questionnaire Prediction History related operations """

    permission_classes = [permissions.IsAuthenticated]

    def get_serializer_class(self):
        language = self.request.query_params.get('language', 'en')
        if language == 'en':
            return DiseaseQuestionnairePredictionSerializer
        elif language == 'si':
            return DiseaseQuestionnairePredictionSinhalaSerializer
        else:
            return DiseaseQuestionnairePredictionSerializer

    def get_queryset(self):
        # get number of records needed from url query params
        limit = self.request.GET.get('limit')

        if limit:
            return DiseaseQuestionnairePrediction.objects.filter(user=self.request.user).order_by('-created_at')[:int(limit)]
        else:
            # if no prediction limit provided return empty
            return DiseaseQuestionnairePrediction.objects.none()

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        language = request.query_params.get('language', None)

        if language == 'si':
            # Set the content type to 'text/plain; charset=utf-8'
            return Response(serializer.data, content_type='text/plain; charset=utf-8')
        else:
            return Response(serializer.data)


