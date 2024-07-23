# Standard library imports
import os
import time
from glob import glob
from pathlib import Path

# Third party imports
import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..georeferencing import georeference
from ..utils import remove_files
from .utils import open_images, save_mask

from ramp.training import (
    callback_constructors,
    loss_constructors,
    metric_constructors,
    model_constructors,
    optimizer_constructors,
)

BATCH_SIZE = 8
IMAGE_SIZE = 256
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class F1_Score(keras.metrics.Metric):
    #  from https://stackoverflow.com/a/64477522
    def __init__(self, class_id, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(class_id=1)
        self.recall_fn = tf.keras.metrics.Recall(class_id=1)

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        self.f1.assign(tf.math.divide_no_nan(2 * (p * r), (p + r)))

    def result(self):
        return self.f1

    def reset_states(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)
    # def get_config(self):
    #     base_config = super(MyLayer, self).get_config()
    #     base_config['output_dim'] = self.output_dim

def get_f1_score_fn():
    return F1_Score(class_id=1)


def predict(
    checkpoint_path: str, input_path: str, prediction_path: str, confidence: float = 0.1
) -> None:
    """Predict building footprints for aerial images given a model checkpoint.

    This function reads the model weights from the checkpoint path and outputs
    predictions in GeoTIF format. The input images have to be in PNG format.

    The predicted masks will be georeferenced with EPSG:3857 as CRS.

    Args:
        checkpoint_path: Path where the weights of the model can be found.
        input_path: Path of the directory where the images are stored.
        prediction_path: Path of the directory where the predicted images will go.
        confidence: Threshold probability for filtering out low-confidence predictions.

    Example::

        predict(
            "model_1_checkpt.tf",
            "data/inputs_v2/4",
            "data/predictions/4"
        )
    """
    start = time.time()
    print(f"Using: {checkpoint_path}")
    # model = keras.models.load_model(checkpoint_path)
    # --- TO_DO: ----
    # changing this to tackle issue with error in loading model:
    # ValueError: Unable to restore custom object of class "F1_Score" (type _tf_keras_metric). Please make sure that this class is included in the `custom_objects` arg when calling `load_model()`. Also, check that the class implements `get_config` and `from_config`
    model = keras.models.load_model(checkpoint_path,
                                    # custom_objects={"f1_scoreee":get_f1_score_fn()},
                                    compile=False)
    # model.compile(
    #             # loss='categorical_crossentropy',
    #               metrics=['categorical_accuracy','f1_score','precision_1', 'recall_1', 'ohe_iou'])
    # # model.evaluate()
    
    # ---
    
    print(f"It took {round(time.time()-start)} sec to load model")
    start = time.time()

    os.makedirs(prediction_path, exist_ok=True)
    print(f'prediction path {prediction_path}')
    image_paths = glob(f"{input_path}/*.tif")
    # print(f'image path {image_paths}')
    for i in range((len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE):
        image_batch = image_paths[BATCH_SIZE * i : BATCH_SIZE * (i + 1)]
        print(f'image batch {image_batch}')
        images = open_images(image_batch)
        images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)

        preds = model.predict(images)
        print(f'one step done ---- CHECK ----- \n __________________________')
        preds = np.argmax(preds, axis=-1)
        preds = np.expand_dims(preds, axis=-1)
        preds = np.where(
            preds > confidence, 1, 0
        )  # Filter out low confidence predictions

        for idx, path in enumerate(image_batch):
            save_mask(
                preds[idx],
                str(f"{prediction_path}/{Path(path).stem}.png"),
            )
    print(
        f"It took {round(time.time()-start)} sec to predict with {confidence} Confidence Threshold"
    )
    keras.backend.clear_session()
    del model
    start = time.time()

    georeference(prediction_path, prediction_path, is_mask=True)
    print(f"It took {round(time.time()-start)} sec to georeference")

    remove_files(f"{prediction_path}/*.xml")
    remove_files(f"{prediction_path}/*.png")
