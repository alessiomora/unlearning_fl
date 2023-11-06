"""Here we verify the accuracy of pretrained ViT on imagenet-1k.

This code is partly adapted from (for ViT/DeiT parts)
https://github.com/sayakpaul/deit-tf/blob/main/i1k_eval/eval-deit.ipynb

We preprocess the dataset with pytorch (resize and crop of images)
because it's difficult to reproduce the exact preprocessing
and results with TF.

Then we store preprocessed images on disk with a folder structure
that permits to load them with TF.

Finally, we load images in a tensorflow dataset, normalize images,
and evaluate two families of models
    - DeiT (ViT-Tiny, ViT-Small here),
    - MiT (MiT-B0, MiT-B1, MiT-B2),

The code can be easily extended to the other models
of the families (e.g., ViT Base, MiT-B3, etc..).
"""
import os

from os import walk

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any {'0', '1', '2'}

from transformers import TFSegformerForImageClassification
import re
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
    test_aircrafts_path = "./aircrafts_test/test"
    train_aircrafts_path = "./aircrafts_test/train"
    generate_imagenet1k_dataset = False
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
        test_aircrafts_path,
        image_size=(256, 256),
        batch_size=None,
        label_mode='int',
    )
    test_ds = (
        test_ds  # .map(cut_lower_20px_fn)
            .map(preprocess_dataset_for_birds_aircafts_cars(is_training=False))
            .map(get_normalization_fn("mit-b0", dataset="aircrafts"))
            .batch(32)
    )
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_aircrafts_path,
        image_size=(256, 256),
        batch_size=None,
        label_mode='int',
    )
    train_ds = (
        train_ds  # .map(cut_lower_20px_fn)
            .map(preprocess_dataset_for_birds_aircafts_cars(is_training=True))
            .map(get_normalization_fn("mit-b0", dataset="aircrafts"))
            .shuffle(2048)
            .batch(32)
    )

    for mit_model in ["nvidia/mit-b0"]:
        print(f"Evaluating {mit_model}.")
        model = get_transformer_model(
                    model_name="mit-b0",
                    classifier_hidden_layers=0,
                    num_classes=100,
                    random_seed=25,
                    load_pretrained_weights=True,
                    trainable_feature_extractor=True,
                    trainable_blocks_fe=None,
        )
        l_w1 = model.get_weights()
        model = get_transformer_model(
                    model_name="mit-b0",
                    classifier_hidden_layers=0,
                    num_classes=100,
                    random_seed=25,
                    load_pretrained_weights=True,
                    trainable_feature_extractor=True,
                    trainable_blocks_fe=1,
        )
        l_w2 = model.get_weights()
        i=0
        for ww in l_w1:
            print(tf.shape(ww))
            print(tf.shape(l_w2[i]))
            print("--")
            i=i+1


        # model.segformer.trainable = False
        # model.classifier.trainable = False
        model.summary(expand_nested=True, show_trainable=True)

        # model.set_weights(model.get_weights())
        optimizer = tf.keras.optimizers.SGD()
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        # model.fit(train_ds.take(3000), epochs=20, validation_data=test_ds)
        # print("[Evaluation]")
        # model.evaluate(test_ds)
        # model.fit(test_ds, epochs=5)
        # print("[Evaluation]")
        # model.evaluate(test_ds)
        # model.fit(test_ds, epochs=5)
        # print("[Evaluation]")
        # model.evaluate(test_ds)
        # model.fit(test_ds, epochs=5)
        # print("[Evaluation]")
        # model.evaluate(test_ds)