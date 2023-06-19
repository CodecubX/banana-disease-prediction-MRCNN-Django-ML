import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from api.models import FertilizerPlan, FertilizerPlanPrediction

from api.serializers.fertilizer_plan import FertilizerPlanSerializer

from api.utils import predict_soil_type
from api.utils.fertilizer_plan import predict
from api.utils.fertilizer_plan.predictor import MODEL_CONFIG

from .utils.utils import build_model_data, get_top_predictions


class FertilizerPlanAPIView(APIView):
    """ Handles Fertilizer Plan Predictions related operations """

    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    serializer_class = FertilizerPlanSerializer

    def get_object(self, *args, **kwargs):
        return FertilizerPlan.objects.get(*args, **kwargs)

    def post(self, request, *args, **kwargs):
        """
        Handles the POST request for fertilizer plan predictions.

        Parameters:
            request (HttpRequest): The HTTP request object.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Response: The HTTP response containing the prediction results and other data.
        """
        data = request.data
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
        original_img_file = default_storage.save('fertilizer_plan/temp_image.jpg',
                                                 ContentFile(file_obj.read()))
        original_img_path = original_img_file

        try:
            # predict soil type
            soil_type = predict_soil_type(os.path.join(settings.MEDIA_ROOT, original_img_path))
            sample_data['soil_type'] = soil_type
            print(f'INFO: Soil type: {soil_type}')
        except Exception as e:
            print(f'ERROR: {e}')
            error = f'Something went wrong while predicting Soil Type'
            return Response({"error": error}, status=status.HTTP_400_BAD_REQUEST)

        try:
            fertilizer_type, _ = predict(sample_data, **MODEL_CONFIG['fertilizer_type'])
            sample_data['fertilizer_type'] = fertilizer_type
            print(f'INFO: {fertilizer_type}')
        except Exception as e:
            print(f'ERROR: {e}')
            error = f'Something went wrong while predicting Fertilizer Type'
            return Response({"error": error}, status=status.HTTP_400_BAD_REQUEST)

        try:
            fertilizer_plan, probabilities = predict(sample_data, **MODEL_CONFIG['fertilizer_plan'])
            sample_data['fertilizer_plan'] = fertilizer_plan
            print(f'INFO: {fertilizer_plan}')
        except Exception as e:
            print(f'ERROR: {e}')
            error = f'Something went wrong while predicting Fertilizer Type'
            return Response({"error": error}, status=status.HTTP_400_BAD_REQUEST)

        # get top 3 predictions
        top_probabilities = get_top_predictions(probabilities)

        context = {
            'prediction': fertilizer_plan,
            'top_probabilities': top_probabilities
        }

        try:
            # retrieve fertilizer plans
            fertilizer_plan = FertilizerPlan.objects.filter(fertilizer_type=fertilizer_type)

            serializer = self.serializer_class(fertilizer_plan, many=True)

            # add fertilizer plan to response data
            context['fertilizer_plan'] = serializer.data

            # import json
            # print(json.dumps(sample_data, indent=4))

            try:
                # Create FertilizerPlanPrediction instance
                fertilizer_plan_prediction_instance = FertilizerPlanPrediction(
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

                    fertilizer_type=self.get_object(fertilizer_type=fertilizer_type),
                    fertilizer_plan=sample_data.get('fertilizer_plan'),
                    top_probabilities=top_probabilities,
                    user=request.user,

                )

                # Save the instance to the database
                fertilizer_plan_prediction_instance.save()
            except Exception as e:
                error = "Failed to save record to history"
                print(f'INFO: Failed to save predictions to database {e}')
                context['error'] = error

        except FertilizerPlan.DoesNotExist:
            error = "Failed to save record to history. Fertilizer Plan does not exists in database"

            context['fertilizer_plan'] = []
            context['error'] = error

        return Response(context, status=status.HTTP_200_OK)



