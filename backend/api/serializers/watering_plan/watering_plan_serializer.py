from rest_framework import serializers

from api.models.watering_plan import WateringPlan


class WateringPlanSerializer(serializers.ModelSerializer):
    """ Serializes the WateringPlan model objects and return selected fields """

    watering_plan = serializers.CharField(source='get_watering_plan_display')
    variety = serializers.StringRelatedField()

    class Meta:
        model = WateringPlan
        fields = '__all__'

    def to_representation(self, instance):
        """Add stage_display field to the serialized representation"""
        representation = super().to_representation(instance)
        representation['stage'] = instance.get_stage_display()
        return representation
