from rest_framework import generics

from api.models.variety_model import Variety

from api.serializers.variety_serializer import VarietySerializer


class VarietyListAPIView(generics.ListAPIView):
    queryset = Variety.objects.all()
    serializer_class = VarietySerializer


class VarietyRetrieveAPIView(generics.RetrieveAPIView):
    queryset = Variety.objects.all()
    serializer_class = VarietySerializer
    lookup_field = 'pk'