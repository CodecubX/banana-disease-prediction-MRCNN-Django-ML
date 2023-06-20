from rest_framework import serializers

from api.models import FertilizerPlanPrediction

from .fertilizer_plan_serializer import FertilizerPlanSerializer


class FertilizerPlanPredictionSerializer(serializers.ModelSerializer):
    """ Serializes the FertilizerPlanPrediction model objects and return selected fields """

    fertilizer_type = FertilizerPlanSerializer()

    class Meta:
        model = FertilizerPlanPrediction
        fields = '__all__'
