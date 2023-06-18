import uuid
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from api.models import Variety, HarvestPrediction

from api.serializers.harvest.harvest_practice_serializer import HarvestPracticeSerializer

from api.utils import predict_soil_type, predict_watering_plan

from .utils.utils import build_model_data


class FertilizerPlanPredictionAPIView(APIView):
    """ Handles Harvest Predictions related operations """

    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    serializer_class = HarvestPracticeSerializer

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
            soil_type = predict_soil_type(original_img_path)
            sample_data['soil_type'] = soil_type
            prediction = predict_watering_plan(sample_data)
        except Exception as e:
            error = str(e).strip('"')
            return Response({"error": error}, status=status.HTTP_400_BAD_REQUEST)

        context = {
            'prediction': prediction,
        }

        # try:
        #     # retrieve practices
        #     variety_obj = Variety.objects.prefetch_related('harvestpractice_set').get(variety=data['variety'])
        #     practices = variety_obj.harvestpractice_set.all()
        #
        #     serializer = self.serializer_class(practices, many=True)
        #
        #     # add post harvest practices to response data
        #     context['post_harvest_practices'] = serializer.data
        #
        #     try:
        #         # Create HarvestPrediction instance
        #         harvest_prediction_instance = HarvestPrediction(
        #             predicted_harvest=prediction,
        #             agro_climatic_region=data['agro_climatic_region'],
        #             plant_density=data['plant_density'],
        #             spacing_between_plants=data['spacing_between_plants'],
        #             pesticides_used=data['pesticides_used'],
        #             plant_generation=data['plant_generation'],
        #             fertilizer_type=data['fertilizer_type'],
        #             soil_pH=data['soil_ph'],
        #             amount_of_sunlight_received=data['amount_of_sunlight'],
        #             watering_schedule=data['watering_schedule'],
        #             number_of_leaves=data['number_of_leaves'],
        #             height=data['height'],
        #             variety=variety_obj,
        #             user=request.user,
        #             harvest=prediction,
        #             top_probabilities=probabilities
        #         )
        #
        #         # Save the HarvestPrediction instance to the database
        #         harvest_prediction_instance.save()
        #     except Exception as e:
        #         print('INFO: Failed to save predictions to database')

        # except Variety.DoesNotExist:
        #     error = "Failed to save record to history. Variety does not exists in database"
        #
        #     context['post_harvest_practices'] = []
        #     context['error'] = error

        return Response(context, status=status.HTTP_200_OK)



