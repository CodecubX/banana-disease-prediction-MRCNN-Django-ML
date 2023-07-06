from rest_framework.generics import ListAPIView
from rest_framework import permissions

from api.models import Cure
from api.serializers import CureSerializer, CureSinhalaSerializer


class CureAPIView(ListAPIView):
    """ Handles retrieving Cures by Disease Id"""

    permission_classes = [permissions.IsAuthenticated]
    serializer_class = CureSerializer

    def get_queryset(self):
        disease_id = self.kwargs.get('disease_id')
        queryset = Cure.objects.filter(disease_id=disease_id)
        return queryset

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context

    def get_serializer_class(self):
        language = self.request.query_params.get('language', None)

        if language == 'si':
            return CureSinhalaSerializer

        return CureSerializer
