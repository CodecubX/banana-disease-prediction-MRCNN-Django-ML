from rest_framework.generics import ListAPIView
from rest_framework import permissions
from rest_framework.response import Response

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

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        language = request.query_params.get('language', None)

        if language == 'si':
            # Set the content type to 'text/plain; charset=utf-8'
            return Response(serializer.data, content_type='text/plain; charset=utf-8')
        else:
            return Response(serializer.data)
