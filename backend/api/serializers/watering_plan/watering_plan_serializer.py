from rest_framework import serializers

from api.models.watering_plan import WateringPlan


class WateringPlanSerializer(serializers.ModelSerializer):
    """ Serializes the WateringPlan model objects and return selected fields """

    class Meta:
        model = WateringPlan
        fields = '__all__'
