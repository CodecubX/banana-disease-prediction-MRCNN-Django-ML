from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from api.models.user_model import User


class RegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[validate_password]
    )
    re_password = serializers.CharField(
        write_only=True,
        required=True
    )

    class Meta:
        model = User
        fields = ('email', 'username', 'password', 're_password', 'first_name', 'last_name')
        extra_kwargs = {'password': {'write_only': True}}

    def validate(self, attrs):
        password = attrs.get('password')
        re_password = attrs.get('re_password')

        if password and re_password and password != re_password:
            raise serializers.ValidationError("The passwords do not match.")

        return attrs

    def create(self, validated_data):
        validated_data.pop('re_password')
        password = validated_data.pop('password', None)

        try:
            validate_password(password)
        except ValidationError as e:
            raise serializers.ValidationError({'password': e.messages})

        instance = self.Meta.model(**validated_data)
        instance.set_password(password)
        instance.save()

        return instance


class UserSerializer(serializers.ModelSerializer):
    profile_pic = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ('id', 'email', 'username', 'first_name', 'last_name', 'profile_pic')

    def get_profile_pic(self, obj):
        if obj.profile_pic:
            # Assuming your profile_pic field stores the file name/path
            # Update the URL format according to your media configuration
            return self.context['request'].build_absolute_uri(obj.profile_pic.url)
        else:
            return None