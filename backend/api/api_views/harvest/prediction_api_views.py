from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response

from api.models import Variety, HarvestPrediction

from api.serializers.harvest.harvest_practice_serializer import HarvestPracticeSerializer

from api.utils import harvest_prediction, calculate_harvesting_time

from .utils.utils import build_model_data


class HarvestPredictionAPIView(APIView):
    """ Handles Harvest Predictions related operations """

    permission_classes = [permissions.IsAuthenticated]

    serializer_class = HarvestPracticeSerializer

    def get(self, request, *args, **kwargs):
        """
        Retrieves the estimated harvesting time based on the variety and age in days.

        Parameters:
            request (HttpRequest): The HTTP request object.
            variety (int): The ID of the variety.
            age (int): The age in days.

        Returns:
            Response: The estimated harvesting time in days as a JSON response.

        Raises:
            NotFound (HTTP 404): If the specified variety is not found.
        """
        # Extract the age in days and variety from the URL params
        variety_id = request.GET.get('variety')
        age_in_days = int(request.GET.get('age'))

        try:
            variety_obj = Variety.objects.get(id=variety_id)
            # Retrieve the average harvesting time from the variety model
            average_harvesting_time = variety_obj.avg_harvesting_time
        except Variety.DoesNotExist:
            return Response({'error': 'Variety not found'}, status=status.HTTP_404_NOT_FOUND)

        # Calculate the estimated harvesting time
        estimated_time = calculate_harvesting_time(average_harvesting_time, age_in_days)

        # Return the estimated harvesting time as a response
        response_data = {
            'estimated_harvesting_time': estimated_time
        }
        return Response(response_data, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        """
        Handles the POST request for harvest prediction.

        Parameters:
            request (HttpRequest): The HTTP request object.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Response: The HTTP response containing the prediction results and other data.
        """
        # build data for model with request data
        sample_data = build_model_data(request.data)

        try:
            # Perform the harvest prediction using your existing function
            prediction, probabilities = harvest_prediction.get_harvest_prediction(sample_data, verbose=True)
        except Exception as e:
            error = str(e).strip('"')
            return Response({"error": error}, status=status.HTTP_400_BAD_REQUEST)

        context = {
            'prediction': prediction,
            'top_probabilities': probabilities
        }

        try:
            # retrieve practices
            variety_obj = Variety.objects.prefetch_related('harvestpractice_set').get(variety=variety)
            practices = variety_obj.harvestpractice_set.all()

            serializer = self.serializer_class(practices, many=True)

            # add post harvest practices to response data
            context['post_harvest_practices'] = serializer.data

            try:
                # Create HarvestPrediction instance
                harvest_prediction_instance = HarvestPrediction(
                    predicted_harvest=prediction,
                    agro_climatic_region=agro_climatic_region,
                    plant_density=plant_density,
                    spacing_between_plants=spacing_between_plants,
                    pesticides_used=pesticides_used,
                    plant_generation=plant_generation,
                    fertilizer_type=fertilizer_type,
                    soil_pH=soil_ph,
                    amount_of_sunlight_received=amount_of_sunlight,
                    watering_schedule=watering_schedule,
                    number_of_leaves=number_of_leaves,
                    height=height,
                    variety=variety_obj,
                    user=request.user,
                    harvest=prediction,
                    top_probabilities=probabilities
                )

                # Save the HarvestPrediction instance to the database
                harvest_prediction_instance.save()
            except Exception as e:
                print('INFO: Failed to save predictions to database')

        except Variety.DoesNotExist:
            error = "Failed to save record to history. Variety does not exists in database"

            context['post_harvest_practices'] = []
            context['error'] = error

        return Response(context, status=status.HTTP_200_OK)



