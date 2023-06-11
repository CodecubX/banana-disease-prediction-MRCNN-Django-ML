import os
import sys
import skimage

from django.conf import settings

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from .mrcnn.config import Config
from .mrcnn.model import MaskRCNN
from .mrcnn.visualize import display_instances

session = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
set_session(session)


# Define the configuration for the model
class ModelConfig(Config):
    NAME = "banana_diseases_coco"
    # number of classes (background + class 1 + class 2)
    NUM_CLASSES = 1 + 3
    # number of training steps per epoch !TEST THIS
    STEPS_PER_EPOCH = 1
    USE_MINI_MASK = False
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8 # Skip detections with < 80% confidence
    CLASS_NAMES = ['BG', 'healthy', 'yellow_leaf', 'anthracnose']


class MaskRCNNModel:
    """
        Wrapper class for the Mask R-CNN model.
    """

    def __init__(self):
        """
        Initializes the Mask R-CNN model and loads the pre-trained weights.
        """
        ROOT_DIR = os.path.abspath("./")
        sys.path.append(ROOT_DIR)
        self.config = ModelConfig()
        self.model = MaskRCNN(mode='inference', model_dir='logs', config=self.config)

        trained_model_path = os.path.join(settings.BASE_DIR, 'api/utils/disease_prediction_mrcnn/saved_models',
                                          f'mask_rcnn_{self.config.NAME}.h5')
        self.model.load_weights(trained_model_path, by_name=True)

        self.class_names = self.config.CLASS_NAMES

    def predict(self, img_path, show_box=False, verbose=False):
        """
        Runs object detection on the specified image and returns the predicted class names.

        Parameters:
        img_path (str): path to the input image

        Returns:
        pred_to_text (list): predicted class names of the detected objects
        """
        # Load the input image
        sample_img = skimage.io.imread(img_path)
        # plt.imshow(sample_img)

        global session
        global graph
        with graph.as_default():
            set_session(session)
            # Perform object detection on the input image
            detected = self.model.detect([sample_img])
            results = detected[0]

            # Display the results of the object detection
            display_instances(
                sample_img,
                results['rois'],
                results['masks'],
                results['class_ids'],
                self.class_names,
                results['scores'],
                "",  # title
                (16, 16),  # figsize
                None,  # ax
                True,  # show_mask=
                True,  # show_mask_polygon
                show_box,  # show_bbox
                None,  # colors
                None,  # captions
                True,  # show_caption
                img_path,  # save the plot
                None,  # filter_classes
                None  # min_score
            )

        # get area of each ROI
        class_scores_areas = {}

        # Iterate over the class IDs, scores, and masks simultaneously
        for class_id, score, mask in zip(results['class_ids'], results['scores'],
                                         results['masks'].transpose(2, 0, 1)):
            # Check if the class ID is already present in the dictionary
            if class_id not in class_scores_areas:
                # If not, initialize an empty list for that class ID
                class_scores_areas[class_id] = []

            # Calculate the area of the mask
            area = mask.sum()

            # Append the score and area to the class's list
            class_scores_areas[class_id].append({'score': score, 'area': area})

        # Convert numerical class IDs to labels
        class_scores_areas_with_labels = {}
        for class_id, scores_areas in class_scores_areas.items():
            class_name = self.class_names[class_id]
            class_scores_areas_with_labels[class_name] = scores_areas

        if verbose:
            print(f'Detected Results:\n\t{class_scores_areas_with_labels}')


        return class_scores_areas_with_labels

# model = MaskRCNNModel()
# pred_to_text = model.predict('bb.jpeg')
# print(pred_to_text)