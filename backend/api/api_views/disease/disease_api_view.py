from rest_framework.generics import RetrieveAPIView, ListAPIView
from rest_framework import permissions

from api.models import Disease
from api.serializers import DiseaseSerializer, DiseaseSinhalaSerializer


class DiseaseAPIView(ListAPIView, RetrieveAPIView):
    """ Handles retrieving all Diseases and one particular Disease by Id """

    permission_classes = [permissions.IsAuthenticated]
    queryset = Disease.objects.all()

    def get_serializer_class(self):
        language = self.request.query_params.get('language', None)

        if language == 'si':
            return DiseaseSinhalaSerializer

        return DiseaseSerializer

    def get(self, request, *args, **kwargs):
        if 'pk' in kwargs:
            # Retrieve a specific object by ID
            return self.retrieve(request, *args, **kwargs)
        else:
            # Get all objects
            return self.list(request, *args, **kwargs)
