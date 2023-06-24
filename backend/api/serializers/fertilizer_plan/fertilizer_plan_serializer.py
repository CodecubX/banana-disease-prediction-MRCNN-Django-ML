from rest_framework import serializers

from api.models.fertilizer_plan import FertilizerPlan


class FertilizerPlanSerializer(serializers.ModelSerializer):
    """ Serializes the FertilizerPlan model objects and return selected fields """

    fertilizer_type = serializers.CharField(source='get_fertilizer_type_display')
    variety = serializers.StringRelatedField()

    class Meta:
        model = FertilizerPlan
        fields = '__all__'

    def to_representation(self, instance):
        """Add stage_display field to the serialized representation"""
        representation = super().to_representation(instance)
        representation['stage'] = instance.get_stage_display()
        return representation
