#################################################################
#
# created July 2024
# Author: Anna Zanchetta
#
#################################################################

import tensorflow as tf
import tensorflow_addons as tfa
# import keras
# from tf.python import keras
# from keras.src import ops

# adding logging
import logging
import functools

log = logging.getLogger()
log.addHandler(logging.NullHandler())

# ---
# --- adding metrics for inference

def sparse_categorical_accuracy_fn():
    return tf.keras.metrics.SparseCategoricalAccuracy()

def categorical_accuracy_fn():
    return tf.keras.metrics.CategoricalAccuracy()

def iou_fn():
    return tf.keras.metrics.IoU(
        num_classes= 2,
        target_class_ids= [0, 1],
        name="iou")

def precision_fn():
    return tf.keras.metrics.Precision(
        # E.g. buildings
        class_id=1,
        name="precision_1")

def recall_fn():
    return tf.keras.metrics.Recall(
        # E.g. buildings
        class_id=1,
        name="recall_1")

def f1score_fn():
    return F1_Score(class_id=1)