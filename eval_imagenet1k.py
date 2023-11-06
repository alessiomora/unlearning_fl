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

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from torchvision import transforms
from PIL import Image

BATCH_SIZE = 32
IMAGE_SIZE = 224

def get_normalization_fn(model_name="mit-b0"):
    if model_name.startswith("mit"):
        transpose = True
    else:
        transpose = False

    def element_norm_imagenet1k_fn(image, label):
        """Normalize input images (imagenet1k)."""
        norm_layer = tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406],
                                                   variance=[np.square(0.229),
                                                             np.square(0.224),
                                                             np.square(0.225)])
        if transpose:
            return tf.transpose(norm_layer(tf.cast(image, tf.float32) / 255.0), (2, 0, 1)), label
        return norm_layer(tf.cast(image, tf.float32) / 255.0), label

    return element_norm_imagenet1k_fn



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
    my_path = os.path.join("imagenet1k", "val")
    data = []
    image_name = []
    dirnames = []

    if generate_imagenet1k_dataset:
        for (dirpath, dirname, filenames) in walk(my_path):
            for filename in filenames:
                data.append(os.path.join(dirpath, filename))
                image_name.append(filename)
                dirnames.append(dirpath.split("/")[2])

        labels = []
        images = []
        cont = 0

        center_crop_fn = tf.keras.layers.CenterCrop(224, 224)

        size = int((256 / 224) * IMAGE_SIZE)
        transform_chain = transforms.Compose(
            [
                transforms.Resize(size, interpolation=3),
                transforms.CenterCrop(IMAGE_SIZE),
            ]
        )

        for d in data:
            img = pil_loader(d)
            to_save_img = transform_chain(img)  # return a PIL Image
            path = os.path.join("imagenet1k_tf", "val", str(dirnames[cont]))
            exist = os.path.exists(path)
            if not exist:
                os.makedirs(path)

            path = os.path.join(path, image_name[cont])
            path = path.replace(".JPEG", ".png")
            print(f"[{cont}] Saving: {path}")
            to_save_img.save(path)
            cont = cont + 1

    val_dir = os.path.join("imagenet1k_tf", "val")
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=(224, 224),  # Define the image size you want
        batch_size=None,
        label_mode='int',  # For multi-class classification problem
    )

    element_norm_imagenet1k_fn = get_normalization_fn("mit-b0")
    val_dataset_mit = (
        val_dataset
        .map(element_norm_imagenet1k_fn)
        .batch(BATCH_SIZE)
    )

    for mit_model in ["nvidia/mit-b0", "nvidia/mit-b1", "nvidia/mit-b2"]:
        print(f"Evaluating {mit_model}.")
        model = TFSegformerForImageClassification.from_pretrained(
            mit_model,
            num_labels=100,
            ignore_mismatched_sizes=True
        )
        # model.segformer.trainable = False
        model.summary(expand_nested=True, show_trainable=True)

        # model.set_weights(model.get_weights())
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.01,
        )
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        model.fit(val_dataset_mit)
        model.evaluate(val_dataset_mit)

    # Expected output
    # Evaluating mit-b0.
    # 196/196 [==============================] - 39s 184ms/step - loss: 5.5205 - accuracy: 0.6927
    # Evaluating mit-b1.
    # 196/196 [==============================] - 39s 184ms/step - loss: 5.2447 - accuracy: 0.7801
    # Evaluating mit-b2.
    # 196/196 [==============================] - 65s 324ms/step - loss: 5.0756 - accuracy: 0.8154

    element_norm_imagenet1k_fn = get_normalization_fn("deit-tiny")
    val_dataset_deit = (
        val_dataset
        .map(element_norm_imagenet1k_fn)
        .batch(BATCH_SIZE)
    )

    # for deit_model in ["deit_tiny", "deit_small"]:
    #     print(f"Evaluating {deit_model}.")
    #     deit_path = model_handle_map[deit_model]
    #     model = get_model(deit_path)
    #     model.summary(expand_nested=True)
    #
    #     model.compile(
    #         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #         metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    #     )
    #     model.evaluate(val_dataset_deit)

    # Expected output
    # Evaluating deit_tiny.
    # 196/196 [==============================] - 39s 184ms/step - loss: 5.5908 - accuracy: 0.7213
    # Evaluating deit_small.
    # 196/196 [==============================] - 65s 324ms/step - loss: 5.3210 - accuracy: 0.7982

    # model = tf.keras.applications.resnet50.ResNet50(
    #     include_top=True,
    #     weights='imagenet',
    #     # classes=100,
    #     classifier_activation='softmax'
    # )
    # model.summary(expand_nested=True)
    #
    # model.compile(
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    # )
    # model.evaluate(val_dataset_deit)
    #
    # model = tf.keras.applications.vgg19.VGG19(
    #     include_top=True,
    #     weights='imagenet',
    #     # classes=100,
    #     classifier_activation='softmax'
    # )
    # model.summary(expand_nested=True)
    # model.compile(
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    # )
    # model.evaluate(val_dataset_deit)

    # print("---- config ----", model.config)
    # inputs = tf.keras.Input((3, 224, 224))
    # _ = model(inputs)
    # model_new = tf.keras.Model(inputs, outputs)
    # model.layers.pop()
    # model.layers.pop()
    # config = model.config
    # config["num_labels"] = 100
    # new_model = TFSegformerForImageClassification(config)
    # new_model(inputs)
    # new_model = tf.keras.models.Sequential(model.layers[:-1])
    # outputs = new_model(inputs)
    #
    # --------------------------------
    #
    # sequence_output = outputs[0]
    #
    # # convert last hidden states to (batch_size, height*width, hidden_size)
    # batch_size = shape_list(sequence_output)[0]
    # sequence_output = tf.transpose(sequence_output, perm=[0, 2, 3, 1])
    # sequence_output = tf.reshape(sequence_output,
    #                              (batch_size, -1, self.config.hidden_sizes[-1]))
    #
    # # global average pooling
    # sequence_output = tf.reduce_mean(sequence_output, axis=1)
    #
    # logits = self.classifier(sequence_output)
    #
    # loss = None if labels is None else self.hf_compute_loss(labels=labels,
    #                                                         logits=logits)
    #
    # if not return_dict:
    #     output = (logits,) + outputs[1:]
    #     return ((loss,) + output) if loss is not None else output
    # --------------------------------
    # model.summary(expand_nested=True)
    # outputs = new_model(inputs)
    # outputs = tf.keras.layers.Dense(1000,
    #                                 kernel_initializer=tf.keras.initializers.RandomNormal(
    #                                     stddev=0.01))(outputs)
    # new_model = tf.keras.Model(inputs, outputs)

