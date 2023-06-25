from rest_framework import serializers

from api.models import Disease

from .cure_serializer import CureSerializer, CureSinhalaSerializer


class DiseaseCureSerializer(serializers.ModelSerializer):
    """ Serializes the Disease model objects """

    cures = CureSerializer(source='cure_set', many=True)  # Nested serialization for associated cures
    img = serializers.SerializerMethodField()
    name_display = serializers.CharField(source='get_name_display', read_only=True)

    class Meta:
        model = Disease
        fields = ['id', 'name', 'name_display', 'img', 'description', 'symptom_description']

    def get_img(self, obj):
        if obj.img:
            return self.context['request'].build_absolute_uri(obj.img.url)
        return None


class DiseaseCureSinhalaSerializer(serializers.ModelSerializer):
    """ Serializes the Disease model objects """

    cures = CureSinhalaSerializer(source='cure_set', many=True)  # Nested serialization for associated cures
    img = serializers.SerializerMethodField()
    name_display = serializers.CharField(source='get_name_display', read_only=True)

    description = serializers.CharField(source='description_sinhala', allow_blank=True)
    symptom_description = serializers.CharField(source='symptom_description_sinhala', allow_blank=True)

    class Meta:
        model = Disease
        fields = ['id', 'name', 'name_display', 'img', 'description', 'symptom_description']

    def get_img(self, obj):
        if obj.img:
            return self.context['request'].build_absolute_uri(obj.img.url)
        return None