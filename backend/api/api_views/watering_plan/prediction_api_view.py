import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from api.models import WateringPlan, WateringPlanPrediction, Variety

from api.serializers.watering_plan import WateringPlanSerializer

from api.utils import predict_soil_type, predict_watering_plan

from .utils.utils import build_model_data, get_top_predictions


class WateringPlanAPIView(APIView):
    """ Handles Watering Plan Predictions related operations """

    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    serializer_class = WateringPlanSerializer

    def get_object(self, *args, **kwargs):
        return WateringPlan.objects.get(*args, **kwargs)

    def post(self, request, *args, **kwargs):
        """
        Handles the POST request for watering plan predictions.

        Parameters:
            request (HttpRequest): The HTTP request object.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Response: The HTTP response containing the prediction results and other data.
        """
        data = request.data
        stage = data.get('stage')
        try:
            variety = int(data.get('variety'))
        except Exception as e:
            print(f'ERROR: {e}')
            return Response(
                {'error': 'Invalid Variety Id. Must be a integer'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # build data for model with request data
        sample_data = build_model_data(data)

        try:
            file_obj = request.FILES['soil_image']
        except Exception:
            return Response(
                {'error': 'Something is wrong with the uploaded file'},
                status=status.HTTP_400_BAD_REQUEST
            )

            # save the image
        original_img_file = default_storage.save('watering_plan/temp_image.jpg',
                                                 ContentFile(file_obj.read()))
        original_img_path = original_img_file

        try:
            # predict soil type
            soil_type = predict_soil_type(os.path.join(settings.MEDIA_ROOT, original_img_path))
            sample_data['soil_type'] = soil_type
        except Exception as e:
            print(f'ERROR: {e}')
            error = 'Something went wrong while predicting Soil Type'
            return Response({"error": error}, status=status.HTTP_400_BAD_REQUEST)

        try:
            prediction, probabilities = predict_watering_plan(sample_data)
        except Exception as e:
            print(f'ERROR: {e}')
            error = f'Something went wrong while predicting Watering Plan'
            return Response({"error": error}, status=status.HTTP_400_BAD_REQUEST)

        top_probabilities = get_top_predictions(probabilities)

        context = {
            'prediction': prediction,
            'top_probabilities': top_probabilities
        }

        try:
            variety = Variety.objects.get(id=variety)
            # retrieve watering plan
            watering_plan = WateringPlan.objects.filter(watering_plan=prediction, variety=variety, stage=stage)

            serializer = self.serializer_class(watering_plan, many=True)

            # add watering plan to response data
            context['watering_plan'] = serializer.data

            try:
                # Create WateringPlanPrediction instance
                watering_plan_prediction_instance = WateringPlanPrediction(
                    pH=sample_data.get('pH'),
                    organic_matter_content=sample_data.get('organic_matter_content'),
                    soil_type=sample_data.get('soil_type'),
                    soil_moisture=sample_data.get('soil_moisture'),
                    avg_temperature=sample_data.get('avg_temperature'),
                    avg_rainfall=sample_data.get('avg_rainfall'),
                    plant_height=sample_data.get('plant_height'),
                    leaf_color=sample_data.get('leaf_color'),
                    stem_diameter=sample_data.get('stem_diameter'),
                    plant_density=sample_data.get('plant_density'),
                    soil_texture=sample_data.get('soil_texture'),
                    soil_color=sample_data.get('soil_color'),
                    temperature=sample_data.get('temperature'),
                    humidity=sample_data.get('humidity'),
                    rainfall=sample_data.get('rainfall'),
                    water_source=sample_data.get('water_source'),
                    irrigation_method=sample_data.get('irrigation_method'),
                    fertilizer_used_last_season=sample_data.get('fertilizer_used_last_season'),
                    crop_rotation=sample_data.get('crop_rotation'),
                    pest_disease_infestation=sample_data.get('pest_disease_infestation'),
                    slope=sample_data.get('slope'),

                    watering_plan=self.get_object(watering_plan=prediction, variety=variety, stage=stage),
                    top_probabilities=top_probabilities,
                    user=request.user,

                )

                # Save the instance to the database
                watering_plan_prediction_instance.save()
            except Exception as e:
                error = "Failed to save record to history"
                print(f'INFO: Failed to save predictions to database {e}')
                context['error'] = error

        except WateringPlan.DoesNotExist:
            error = "Failed to save record to history. Watering Plan does not exists in database"

            context['watering_plan'] = []
            context['error'] = error

        return Response(context, status=status.HTTP_200_OK)



