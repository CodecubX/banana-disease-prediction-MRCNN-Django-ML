from rest_framework import serializers

from api.models.watering_plan import WateringPlan


class WateringPlanSerializer(serializers.ModelSerializer):
    """ Serializes the WateringPlan model objects and return selected fields """

    watering_plan = serializers.CharField(source='get_watering_plan_display')
    stage = serializers.CharField(source='get_stage_display')
    variety = serializers.StringRelatedField()

    class Meta:
        model = WateringPlan
        fields = '__all__'
