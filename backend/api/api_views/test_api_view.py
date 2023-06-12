from rest_framework import permissions
from rest_framework.generics import ListAPIView
from rest_framework import serializers

from api.models import Test


class TestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Test
        fields = '__all__'


class TestAPIView(ListAPIView):
    """ For testing purposes only """

    permission_classes = [permissions.IsAuthenticated]

    queryset = Test.objects.all()
    serializer_class = TestSerializer









