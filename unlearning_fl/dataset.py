"""Handle dataset loading and preprocessing utility."""
import os
from typing import Union

import numpy as np
import tensorflow as tf


def load_selected_client_statistics(
        selected_client: int,
        alpha: float,
        dataset: str,
        total_clients: int,
):
    """Return the amount of local examples for the selected client.

    Clients are referenced with a client_id. Loads a numpy array saved on disk. This
    could be done directly by doing len(ds.to_list()) but it's more expensive at run
    time.
    """
    if alpha < 0:
        alpha_dirichlet_string = "iid"
    else:
        alpha_dirichlet_string = str(round(alpha, 2))
    path = os.path.join(
        "federated_datasets",
        dataset + "_dirichlet",
        str(total_clients),
        alpha_dirichlet_string,
        "distribution_train.npy",
    )
    smpls_loaded = np.load(path)
    # tf.print(smpls_loaded, summarize=-1)
    local_examples_all_clients = np.sum(smpls_loaded, axis=1)
    return local_examples_all_clients[selected_client]


# pylint: disable=W0221
class PaddedRandomCropCustom(tf.keras.layers.Layer):
    """Custom keras layer to random crop the input image, same as FedMLB paper."""
    def __init__(
            self, seed: Union[int, None] = None, height: int = 32, width: int = 32,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.seed = seed
        self.height = height
        self.width = width

    def call(self, inputs: tf.Tensor, training: bool = True):
        """Call the layer on new inputs and returns the outputs as tensors."""
        if training:
            inputs = tf.image.resize_with_crop_or_pad(
                image=inputs,
                target_height=self.height + 4,
                target_width=self.width + 4,
            )
            inputs = tf.image.random_crop(
                value=inputs, size=[self.hfeight, self.width, 3], seed=self.seed
            )

            return inputs
        return inputs


# pylint: disable=W0221
class PaddedCenterCropCustom(tf.keras.layers.Layer):
    """Custom keras layer to center crop the input image, same as FedMLB paper."""
    def __init__(self, height: int = 64, width: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width

    def call(self, inputs: tf.Tensor):
        """Call the layer on new inputs and returns the outputs as tensors."""
        input_tensor = tf.image.resize_with_crop_or_pad(
            image=inputs, target_height=self.height, target_width=self.width
        )

        input_shape = tf.shape(inputs)
        h_diff = input_shape[0] - self.height
        w_diff = input_shape[1] - self.width

        h_start = tf.cast(h_diff / 2, tf.int32)
        w_start = tf.cast(w_diff / 2, tf.int32)
        return tf.image.crop_to_bounding_box(
            input_tensor, h_start, w_start, self.height, self.width
        )


# pylint: disable=W0221
class RandomResizedCrop(tf.keras.layers.Layer):
    """Preprocessing birds, aircrafts, cars."""
    def __init__(self, size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), **kwargs):
        super().__init__(**kwargs)
        self.crop_shape = size
        self.scale = scale
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

    def call(self, inputs: tf.Tensor):
        """Call the layer on new inputs and returns the outputs as tensors."""

        inputs = tf.expand_dims(inputs, axis=0)
        batch_size = tf.shape(inputs)[0]
        # tf.print("batch_size ", batch_size)
        random_scales = tf.random.uniform(
            (batch_size,),
            self.scale[0],
            self.scale[1]
        )
        random_ratios = tf.exp(tf.random.uniform(
            (batch_size,),
            self.log_ratio[0],
            self.log_ratio[1]
        ))

        new_heights = tf.clip_by_value(
            tf.sqrt(random_scales / random_ratios),
            0,
            1,
        )
        new_widths = tf.clip_by_value(
            tf.sqrt(random_scales * random_ratios),
            0,
            1,
        )
        height_offsets = tf.random.uniform(
            (batch_size,),
            0,
            1 - new_heights,
        )
        width_offsets = tf.random.uniform(
            (batch_size,),
            0,
            1 - new_widths,
        )

        bounding_boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + new_heights,
                width_offsets + new_widths,
            ],
            axis=1,
        )
        images = tf.image.crop_and_resize(
            inputs,
            bounding_boxes,
            tf.range(batch_size),
            self.crop_shape,
        )
        image = tf.squeeze(images, axis=0)
        # tf.print("image ", tf.shape(image))
        return image


def get_normalization_fn(model_name="mit-b0", dataset="cifar100"):
    """Return the normalization function based on model family and dataset."""
    if model_name.startswith("mit"):
        transpose = True
    else:
        transpose = False

    if dataset in ["cifar100"]:
        mean = [0.5071, 0.4865, 0.4409]
        variance = [np.square(0.2673), np.square(0.2564), np.square(0.2762)]
    elif dataset in ["aircrafts"]:
        mean = [0.4862, 0.5179, 0.5420]
        variance = [np.square(0.1920), np.square(0.1899), np.square(0.2131)]
    elif dataset in ["cars"]:
        mean = [0.4668, 0.4599, 0.4505]
        variance = [np.square(0.2642), np.square(0.2636), np.square(0.2687)]
    else:
        mean = [0.485, 0.456, 0.406]
        variance = [np.square(0.229), np.square(0.224), np.square(0.225)]

    def element_norm_fn(image, label):
        """Normalize cifar100 images."""
        norm_layer = tf.keras.layers.Normalization(
            mean=mean,
            variance=variance,
        )
        if transpose:
            return tf.transpose(norm_layer(tf.cast(image, tf.float32) / 255.0),
                                    (2, 0, 1)), label

        return norm_layer(tf.cast(image, tf.float32) / 255.0), label

    return element_norm_fn


def load_client_datasets_from_files(  # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        dataset: str,
        sampled_client: int,
        batch_size: int,
        total_clients: int = 100,
        alpha: float = 0.3,
        split: str = "train",
        model_name: str = "mit-b0",
        is_training: bool = True,
        seed: Union[int, None] = None,
):
    """Load the partition of the dataset for the sampled client.

    Sampled client represented by its client_id. Examples are preprocessed via
    normalization layer. Returns a batched dataset.
    """

    if alpha < 0:
        alpha_dirichlet_string = "iid"
    else:
        alpha_dirichlet_string = str(round(alpha, 2))
    path = os.path.join(
        "federated_datasets",
        dataset + "_dirichlet",
        str(total_clients),
        alpha_dirichlet_string,
        split,
    )

    loaded_ds = tf.data.Dataset.load(
        path=os.path.join(path, str(sampled_client)),
        element_spec=None,
        compression=None,
        reader_func=None,
    )

    if dataset in ["cifar100"]:
        loaded_ds = (
            loaded_ds.map(preprocess_dataset_for_transformers_models(is_training))
                .map(get_normalization_fn(model_name=model_name))
                .shuffle(2048)
                .batch(batch_size, drop_remainder=False)
        )
        loaded_ds = loaded_ds.prefetch(tf.data.AUTOTUNE)
        return loaded_ds
    elif dataset in ["birds", "aircrafts", "cars"]:
        loaded_ds = (
            loaded_ds.map(preprocess_dataset_for_birds_aircafts_cars(is_training))
                .map(get_normalization_fn(model_name=model_name, dataset=dataset))
                .shuffle(2048)
                .batch(batch_size, drop_remainder=False)
        )
        loaded_ds = loaded_ds.prefetch(tf.data.AUTOTUNE)
        return loaded_ds

    # else
    loaded_ds = loaded_ds.prefetch(tf.data.AUTOTUNE)
    return loaded_ds


def preprocess_dataset_for_transformers_models(is_training=True, resolution=224):
    def resize_and_crop_fn(image, label):
        if is_training:
            # Resize to a bigger spatial resolution and take the random
            # crops.
            image = tf.image.resize(image, (resolution + 20, resolution + 20))
            image = tf.image.random_crop(image, (resolution, resolution, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (resolution, resolution))
        return image, label

    return resize_and_crop_fn


def preprocess_dataset_for_birds_aircafts_cars(is_training=True, resolution=224):
    def resize_and_crop_fn(image, label):
        if is_training:
            # Resize to a bigger spatial resolution and take the random
            # crops.
            random_resize_crop_fn = RandomResizedCrop(size=(resolution, resolution))
            image = random_resize_crop_fn(image)
            image = tf.image.random_flip_left_right(image)
            # image = tf.image.resize(image, (256, 256))
            # image = tf.keras.layers.CenterCrop(resolution, resolution)(image)
        else:
            # image = tf.image.resize(image, (256, 256))
            image = tf.keras.layers.CenterCrop(resolution, resolution)(image)
        return image, label

    return resize_and_crop_fn