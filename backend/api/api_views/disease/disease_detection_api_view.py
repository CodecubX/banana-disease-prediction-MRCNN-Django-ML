import os
import shutil
import uuid
import json
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework import permissions, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from api.models import Disease, DiseasePrediction

from api.utils.disease_prediction_mrcnn.predictor import MaskRCNNModel

from api.serializers import DiseaseCureSerializer

from .utils.utils import filter_and_calculate_area

model = MaskRCNNModel()


class BananaDiseaseMRCNNAPIView(APIView):
    """ Handles Disease Detection using MRCNN model and related operations """

    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    serializer_class = DiseaseCureSerializer

    def get_object(self, *args, **kwargs):
        """
        Retrieves a Disease object based on the given filter parameters.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Disease: The Disease object matching the filter parameters.
        """
        return Disease.objects.get(*args, **kwargs)

    def post(self, request, *args, **kwargs):
        """
        Handles the POST request for disease detection and related operations.

        Parameters:
            request (HttpRequest): The HTTP request object.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Response: The HTTP response containing the detection results and related information.
        """
        try:
            file_obj = request.FILES['image']
        except Exception:
            return Response(
                {'error': 'Something is wrong with the uploaded file'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # save the image
        original_img_file = default_storage.save('disease_detection/original/temp_image.jpg', ContentFile(file_obj.read()))
        original_img_path = original_img_file

        destination_directory = 'disease_detection/detected'
        # create unique file name
        unique_filename = str(uuid.uuid4())[:20] + '.jpg'

        destination_path = os.path.join(destination_directory, unique_filename)

        # create the destination directory if it doesn't exist
        os.makedirs(destination_directory, exist_ok=True)
        # save copy
        shutil.copyfile(os.path.join(settings.MEDIA_ROOT, original_img_path), os.path.join(settings.MEDIA_ROOT,destination_path))

        try:
            # get predictions
            predictions = model.predict(os.path.join(settings.MEDIA_ROOT, destination_path))
        except Exception as e:
            print(f'INFO: {e}')
            return Response({'error': 'Something went wrong while detecting'}, status=status.HTTP_409_CONFLICT)

        # Construct the URLs for the output images
        original_img_url = request.build_absolute_uri(settings.MEDIA_URL + original_img_path)
        detected_img_url = request.build_absolute_uri(settings.MEDIA_URL + destination_path)

        # Sort the result array based on total area in descending order
        sorted_results = filter_and_calculate_area(predictions)

        top_disease = sorted_results[0]['disease_name']

        context = {
            'probabilities': sorted_results,
            'original_img_url': original_img_url,
            'detected_img_url': detected_img_url,
        }

        try:
            disease = self.get_object(name=top_disease)
            serializer = self.serializer_class(disease, context={'request': request})
            disease_data = serializer.data

            try:
                # Create an instance of DiseasePrediction
                prediction = DiseasePrediction()

                # Set the fields with the corresponding values
                prediction.img = original_img_path
                prediction.detected_img = destination_path
                prediction.user = request.user
                prediction.disease = disease
                prediction.probabilities = sorted_results
                # Save the instance
                prediction.save()
                print(f'INFO: Detection data saved to history')
            except Exception as e:
                error = 'Failed to save to history.'
                print(f'INFO: Failed to save to history: {e}')
                context['error'] = error

            # add disease and cure data to response data
            context['disease'] = disease_data
        except Disease.DoesNotExist:
            error = 'Disease not found'
            context['disease'] = []
            context['error'] = error

        return Response(context)
