import os

from unlearning_fl.vit import ViTDistilled, ViTClassifier
from unlearning_fl.vit.model_configs import base_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from transformers import TFSegformerForImageClassification

tf.get_logger().setLevel(logging.ERROR)


RESOLUTION = 224
NUM_CLASSES = 100
TEST_BATCH_SIZE = 32

model_handle_map = {
    "deit_tiny": "https://tfhub.dev/sayakpaul/deit_tiny_patch16_224/1",
    "deit_tiny_distilled": "https://tfhub.dev/sayakpaul/deit_tiny_distilled_patch16_224/1",
    "deit_small": "https://tfhub.dev/sayakpaul/deit_small_patch16_224/1",
    "deit_small_distilled": "https://tfhub.dev/sayakpaul/deit_small_distilled_patch16_224/1",
    "deit_base": "https://tfhub.dev/sayakpaul/deit_base_patch16_224/1",
    "deit_base_distilled": "https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_224/1",
    "deit_base_patch16_384": "https://tfhub.dev/sayakpaul/deit_base_patch16_384/1",
    "deit_base_distilled_patch16_384": "https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_384/1",
    "mit-b0": "nvidia/mit-b0",
    "mit-b1": "nvidia/mit-b1",
    "mit-b2": "nvidia/mit-b2",
}


def set_non_trainable_blocks(model, non_trainable_blocks=0):
    """Set blocks of model to non trainable starting by the deeper ones."""
    non_trainable_blocks_list = range(4-non_trainable_blocks)

    for layer in model.get_layer('segformer').encoder.embeddings:
        layer.trainable = True
        for i in non_trainable_blocks_list:
            if layer.name.startswith("patch_embeddings." + str(i)):
                # print(layer.name)
                layer.trainable = False

    for layer in model.get_layer('segformer').encoder.layer_norms:
        layer.trainable = True
        for i in non_trainable_blocks_list:
            if layer.name.startswith("layer_norm." + str(i)):
                # print(layer.name)
                layer.trainable = False

    for layer in model.get_layer('segformer').encoder.block:
        for i in non_trainable_blocks_list:
            for l in layer:
                l.trainable = True
                if l.name.startswith("block." + str(i)):
                    # print(l.name)
                    l.trainable = False

    return model


def attach_classifier_head(inputs, outputs, num_classes, classifier_hidden_layers, classifier_unit_pl, random_seed=23):
    for _ in range(0, classifier_hidden_layers):
        outputs = tf.keras.layers.Dense(classifier_unit_pl,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(
                                            stddev=0.01, seed=random_seed))(outputs)
        outputs = tf.keras.layers.BatchNormalization(momentum=1e-5)(outputs)
        outputs = tf.keras.layers.ReLU()(outputs)

    outputs = tf.keras.layers.Dense(num_classes,
                                    kernel_initializer=tf.keras.initializers.RandomNormal(
                                        stddev=0.01, seed=random_seed))(outputs)
    return tf.keras.Model(inputs, outputs)


def get_transformer_model(
        load_pretrained_weights: bool = True,
        model_name: str = "deit_tiny",
        res: int = RESOLUTION,
        num_classes: int = NUM_CLASSES,
        trainable_feature_extractor: bool = True,
        classifier_hidden_layers: int = 0,
        image_resolution: int = 224,
        classifier_unit_pl: int = 1000,
        random_seed: int = 21,
        trainable_blocks_fe: int = None,
) -> tf.keras.Model:
    """Return the requested pre-trained deit or mit model with a randomly initialized
    classifier. """

    if load_pretrained_weights:
        print(f"Loading pretrained extractor for {model_name}.")
        model_gcs_path = model_handle_map[model_name]
        if model_name.startswith("deit"):
            inputs = tf.keras.Input((res, res, 3))
            hub_module = hub.KerasLayer(
                model_gcs_path,
                trainable=trainable_feature_extractor
            )

            # Second output in the tuple is a dictionary containing attention scores.
            outputs, _ = hub_module(inputs)
            model = attach_classifier_head(inputs, outputs, num_classes, classifier_hidden_layers, classifier_unit_pl, random_seed=random_seed)
        else:  # mit
            model = TFSegformerForImageClassification.from_pretrained(
                model_gcs_path,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            if not trainable_feature_extractor:
                model.segformer.trainable = trainable_feature_extractor
                return model

            if trainable_blocks_fe is None:
                return model

            model = set_non_trainable_blocks(model, trainable_blocks_fe)

        return model

    print(f"Initializing {model_name} from scratch.")
    # only implemented for vit/deit
    extractor_config = base_config.get_config(
        model_name=model_name+"_patch16_224"
    )
    extractor = ViTClassifier(extractor_config)
    extractor.trainable = trainable_feature_extractor
    inputs = tf.keras.Input((image_resolution, image_resolution, 3))
    outputs, _ = extractor(inputs)
    model = attach_classifier_head(inputs, outputs, num_classes, classifier_hidden_layers, classifier_unit_pl, random_seed=random_seed)

    return model


# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# test_ds = test_ds.map(element_norm_fn_cifar100).map(preprocess_dataset()).batch(
#     TEST_BATCH_SIZE)
#
# train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_ds_cardinality = train_ds.cardinality().numpy()
# train_instances = int(train_ds_cardinality/100*90)
#
# validation_ds = train_ds.skip(train_instances).map(element_norm_fn_cifar100).map(preprocess_dataset()).batch(
#     TEST_BATCH_SIZE)
# train_ds = train_ds.take(train_instances).shuffle(1024).map(element_norm_fn_cifar100).map(preprocess_dataset()).batch(
#     TEST_BATCH_SIZE)