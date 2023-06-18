from rest_framework import serializers

from api.models.watering_plan import WateringPlanPrediction

from .watering_plan_serializer import WateringPlanSerializer


class WateringPlanPredictionSerializer(serializers.ModelSerializer):
    """ Serializes the WateringPlanPrediction model objects and return selected fields """

    watering_plan = WateringPlanSerializer(many=True)

    class Meta:
        model = WateringPlanPrediction
        fields = '__all__'
