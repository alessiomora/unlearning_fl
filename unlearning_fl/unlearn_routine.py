"""Client unlearn routine
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
    client_model = get_transformer_model(
        model_name=model_name,
        classifier_hidden_layers=classifier_hidden_layers,
        num_classes=num_classes,
        random_seed=random_seed,
        load_pretrained_weights=load_pretrained_weights,
        trainable_feature_extractor=trainable_feature_extractor,
        trainable_blocks_fe=trainable_blocks_fe
    )

    path = os.path.join(save_path_checkpoints, "dict_info.pickle")
    last_checkpoint = dic_load(path)["checkpoint_round"]
    if last_checkpoint:
    print(f"Loading saved checkpoint round {last_checkpoint}")
    path = os.path.join(
            save_path_checkpoints,
            "checkpoints_R" + str(last_checkpoint),
            "server_model",
    )
    client_model.load_weights(path)

    test_ds = tfds.load("caltech_birds2011", split='test',
                        shuffle_files=False, as_supervised=True)
    test_ds = (
        test_ds.map(
            preprocess_dataset_for_birds_aircafts_cars(is_training=False))
            .map(get_normalization_fn(model_name, dataset="birds"))
            .batch(TEST_BATCH_SIZE)
    )

    train_ds = load_client_datasets_from_files(
        dataset=dataset,
        sampled_client=int(cid),
        total_clients = total_clients,
        batch_size = local_batch_size,
        alpha = alpha_dirichlet,
        seed = random_seed,
    )

    optimizer = tf.keras.optimizers.SGD()
    client_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    results = client_model.fit(train_ds, epochs=epochs, verbose=0)
    client_model.evaluate(test_ds)