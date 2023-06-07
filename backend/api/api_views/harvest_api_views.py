from rest_framework import status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response

from api.utils import harvest_prediction


class HarvestPredictionAPIView(APIView):
    """ Handles Harvest Predictions related operations """

    permission_classes = [permissions.IsAuthenticated]

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

        # Return the prediction and probabilities as a JSON response
        response_data = {
            'prediction': prediction,
            'probabilities': probabilities
        }
        return Response(response_data)



