"""
"""
import os

from os import walk

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any {'0', '1', '2'}

from transformers import TFSegformerForImageClassification

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from torchvision import transforms
from PIL import Image
from unlearning_fl.model_utility import get_transformer_model
from unlearning_fl.dataset import (
    get_normalization_fn,
    preprocess_dataset_for_birds_aircafts_cars,
)


BATCH_SIZE = 32
IMAGE_SIZE = 224


def get_model(model_url):
    """Load a checkpoint model from its url."""
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    hub_module = hub.KerasLayer(model_url)

    outputs, _ = hub_module(inputs)

    return tf.keras.Model(inputs, outputs)


def pil_loader(path):
    """Load a PIL image."""
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


if __name__ == '__main__':
    train_cars_path = './stanford-car-dataset-by-classes-folder/car_data/car_data/train'
    test_cars_path = './stanford-car-dataset-by-classes-folder/car_data/car_data/test'
    model_handle_map = {
        "deit_tiny": "https://tfhub.dev/sayakpaul/deit_tiny_patch16_224/1",
        "deit_tiny_distilled": "https://tfhub.dev/sayakpaul/deit_tiny_distilled_patch16_224/1",
        "deit_small": "https://tfhub.dev/sayakpaul/deit_small_patch16_224/1",
        "deit_small_distilled": "https://tfhub.dev/sayakpaul/deit_small_distilled_patch16_224/1",
        "deit_base": "https://tfhub.dev/sayakpaul/deit_base_patch16_224/1",
        "deit_base_distilled": "https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_224/1",
        "deit_base_patch16_384": "https://tfhub.dev/sayakpaul/deit_base_patch16_384/1",
        "deit_base_distilled_patch16_384": "https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_384/1",
    }

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_cars_path,
        image_size=(256, 256),
        batch_size=None,
        label_mode='int',
    )
    test_ds = (
        test_ds
            .map(preprocess_dataset_for_birds_aircafts_cars(is_training=False))
            .map(get_normalization_fn("mit-b0", dataset="cars"))
            .batch(32)
    )
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_cars_path,
        image_size=(256, 256),
        batch_size=None,
        label_mode='int',
    )
    train_ds = (
        train_ds
            .map(preprocess_dataset_for_birds_aircafts_cars(is_training=True))
            .map(get_normalization_fn("mit-b0", dataset="cars"))
            .shuffle(2048)
            .batch(32)
    )

    for mit_model in ["nvidia/mit-b0"]:
        print(f"Evaluating {mit_model}.")
        model = get_transformer_model(
                    model_name="mit-b0",
                    classifier_hidden_layers=0,
                    num_classes=196,
                    random_seed=25,
                    load_pretrained_weights=True,
                    trainable_feature_extractor=True
                )
        # model.segformer.trainable = False
        model.summary(expand_nested=True, show_trainable=True)

        # model.set_weights(model.get_weights())
        # optimizer = tf.keras.optimizers.SGD()  # 0.35, 0.81
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=3e-4,
            clipnorm=10.0,
            weight_decay=1e-3,
        )
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        model.fit(train_ds, epochs=50, validation_data=test_ds)
        print("[Evaluation]")
        model.evaluate(test_ds)
        # model.fit(test_ds, epochs=5)
        # print("[Evaluation]")
        # model.evaluate(test_ds)
        # model.fit(test_ds, epochs=5)
        # print("[Evaluation]")
        # model.evaluate(test_ds)
        # model.fit(test_ds, epochs=5)
        # print("[Evaluation]")
        # model.evaluate(test_ds)