from rest_framework import permissions, status
from rest_framework.views import APIView
from rest_framework.response import Response

from api.models import Disease
from api.serializers import DiseaseSerializer, DiseaseSinhalaSerializer


class DiseaseAPIView(APIView):
    """ Handles retrieving all Diseases and one particular Disease by Id """

    permission_classes = [permissions.IsAuthenticated]

    def get_serializer_class(self):
        language = self.request.query_params.get('language', None)

        if language == 'si':
            return DiseaseSinhalaSerializer

        return DiseaseSerializer

    def get(self, request, *args, **kwargs):
        serializer_class = self.get_serializer_class()

        language = request.query_params.get('language', None)
        disease_id = kwargs.get('pk')
        if disease_id is not None:
            # Retrieve a specific object by ID
            try:
                disease = Disease.objects.get(id=disease_id)
                serializer = serializer_class(disease)
            except Disease.DoesNotExist:
                return Response({'error': 'Disease not found'}, status=status.HTTP_404_NOT_FOUND)
        else:
            # get all objects
            diseases = Disease.objects.all()
            serializer = serializer_class(diseases, many=True)

        if language == 'si':
            return Response(serializer.data, status=status.HTTP_200_OK, content_type='text/plain; charset=utf-8')

        return Response(serializer.data, status=status.HTTP_200_OK)
