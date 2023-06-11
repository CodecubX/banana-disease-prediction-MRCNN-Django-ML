from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response

from api.models import Variety, HarvestPrediction

from api.serializers.harvest.harvest_practice_serializer import HarvestPracticeSerializer

from api.utils import harvest_prediction, calculate_harvesting_time


class HarvestPredictionAPIView(APIView):
    """ Handles Harvest Predictions related operations """

    permission_classes = [permissions.IsAuthenticated]

    serializer_class = HarvestPracticeSerializer

    def get(self, request, *args, **kwargs):
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
        # Extract the data from the request
        variety = request.data.get('variety')
        agro_climatic_region = request.data.get('agro_climatic_region')
        plant_density = request.data.get('plant_density')
        spacing_between_plants = request.data.get('spacing_between_plants')
        pesticides_used = request.data.get('pesticides_used')
        plant_generation = request.data.get('plant_generation')
        fertilizer_type = request.data.get('fertilizer_type')
        soil_ph = request.data.get('soil_ph')
        amount_of_sunlight = request.data.get('amount_of_sunlight')
        watering_schedule = request.data.get('watering_schedule')
        number_of_leaves = request.data.get('number_of_leaves')
        height = request.data.get('height')

        # Create the data dictionary
        sample_data = {
            "Variety": variety,
            "Agro-climatic region": agro_climatic_region,
            "Plant density(Min=1,Max=5)": plant_density,
            "Spacing between plants (m)": spacing_between_plants,
            'Pesticides used(Yes, No)': pesticides_used,
            "Plant generation": plant_generation,
            "Fertilizer type": fertilizer_type,
            "Soil pH": soil_ph,
            "Amount of sunlight received": amount_of_sunlight,
            "Watering schedule": watering_schedule,
            "Number of leaves": number_of_leaves,
            "Height (m)": height
        }

        try:
            # Perform the harvest prediction using your existing function
            prediction, probabilities = harvest_prediction.get_harvest_prediction(sample_data, verbose=True)
        except Exception as e:
            error = str(e).strip('"')
            return Response({"error": error}, status=status.HTTP_400_BAD_REQUEST)

        # retrieve practices
        variety_obj = Variety.objects.prefetch_related('harvestpractice_set').get(variety=variety)
        practices = variety_obj.harvestpractice_set.all()

        serializer = self.serializer_class(practices, many=True)

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

        # Return the prediction and probabilities as a JSON response
        response_data = {
            'prediction': prediction,
            'top_probabilities': probabilities,
            'post_harvest_practices': serializer.data
        }
        return Response(response_data, status=status.HTTP_200_OK)



