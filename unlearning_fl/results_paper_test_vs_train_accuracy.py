"""Assessing the performance of global models and sanitized global models
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any {'0', '1', '2'}

import requests
import shutil
import tarfile
from tqdm import tqdm
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
from huggingface_hub import snapshot_download

from unlearning_fl.dataset import (
    get_normalization_fn,
    preprocess_dataset_for_birds_aircafts_cars,
    load_client_datasets_from_files,
    preprocess_dataset_for_transformers_models,
)
from unlearning_fl.model_utility import get_transformer_model
from unlearning_fl.prepare_aircraft_dataset import (
    download_aircracft_dataset,

)


TEST_AIRCRAFTS_DATASET = "./aircrafts_test/test"
TEST_BATCH_SIZE = 128

TABLE_DATASET_CLASSES = {
    "cifar100": 100,
    "birds": 200,
    "cars": 196,
    "aircrafts": 100
}


def pil_loader(path):
    """Load a PIL image."""
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def cut_lower_20px_fn(image):
    target_height = tf.shape(image)[0] - 20
    target_width = tf.shape(image)[1]
    img = tf.image.crop_to_bounding_box(
        image, offset_height=0, offset_width=0, target_height=target_height,
        target_width=target_width
    )
    return img


def download_aircracft_dataset():
    tar_url = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
    fname = "fgvc-aircraft-2013b.tar.gz"
    chunk_size = 1024

    resp = requests.get(tar_url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

    if tarfile.is_tarfile(fname):
        with tarfile.open(fname) as f:
            f.extractall(path="fgvc-aircraft")
    return


def get_aircrafts_test_set():
    data_path_img = "./fgvc-aircraft/fgvc-aircraft-2013b/data/images"
    data_path = "./fgvc-aircraft/fgvc-aircraft-2013b/data"
    test_path = os.path.join(data_path, "images_variant_test.txt")
    variants_label_path = os.path.join(data_path,
                                       "variants.txt")  # contains all the labels
    convert_label_to_index_dict = {}
    variant_label_id = 0

    with open(variants_label_path) as variants_file_txt:
        for line in variants_file_txt:
            variant_as_string = line.split("\n")[0]  # remove the ending line
            # print(variant_as_string)
            convert_label_to_index_dict[variant_as_string] = variant_label_id
            variant_label_id = variant_label_id + 1
    cont = 0
    alphabetical_order = [str(i) for i in range(0, 100)]
    alphabetical_order.sort()
    with open(test_path) as test_file_txt:
        for line in test_file_txt:
            split = line.split(maxsplit=1)
            img_file_name = split[0]
            label = convert_label_to_index_dict[split[1].split("\n")[0]]
            folder_name = alphabetical_order[label]
            img_path = os.path.join(data_path_img, img_file_name + ".jpg")
            img = pil_loader(img_path)

            to_save_img = cut_lower_20px_fn(img)
            # path = os.path.join("aircrafts_test", "test", str(label))
            path = os.path.join("aircrafts_test", "test", folder_name)
            exist = os.path.exists(path)
            if not exist:
                os.makedirs(path)

            to_save_img = Image.fromarray(np.uint8(to_save_img.numpy()))
            path = os.path.join(path, img_file_name + ".jpg")
            path = path.replace(".jpg", ".png")
            # img.save('test.jpg', quality="keep")
            shutil.copy(img_path, path)
            print(f"[{cont}] Saving: {path}")
            cont = cont + 1
            to_save_img.save(path)

    return


def return_test_ds(dataset, model_name="mit-b0"):
    if dataset in ["cifar100"]:
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_ds = (
            test_ds.map(
                preprocess_dataset_for_transformers_models(is_training=False))
                .map(get_normalization_fn(model_name))
                .batch(TEST_BATCH_SIZE)
        )
        return test_ds
    elif dataset in ["birds"]:
        test_ds = tfds.load("caltech_birds2011", split='test',
                            shuffle_files=False, as_supervised=True)
    elif dataset in ["aircrafts"]:
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            TEST_AIRCRAFTS_DATASET,
            image_size=(256, 256),
            batch_size=None,
            label_mode='int',
        )

    test_ds = (
        test_ds.map(
            preprocess_dataset_for_birds_aircafts_cars(is_training=False))
            .map(get_normalization_fn(model_name, dataset=dataset))
            .batch(TEST_BATCH_SIZE)
    )

    return test_ds

def main() -> None:
    datasets = ["cifar100", "birds", "aircrafts"]
    alpha_dirichlet = -1  # iid
    total_clients_in_ds = [100, 29, 65]  # total number of clients in the federation
    clients_per_round_in_ds = [10, 5, 7]
    model_name = "mit-b0"  # mit-b0, mit-b1, mit-b2, deit_tiny, deit_small
    classifier_hidden_layers = 0
    load_pretrained_weights = True
    trainable_feature_extractor = True
    trainable_blocks_fe = None
    batch_size = 32
    random_seed = 25
    cid = 0

    if not os.path.exists(path="./fgvc-aircraft/"):
        print("[Downloading Datasets...]")
        download_aircracft_dataset()
    if not os.path.exists(path="./aircrafts_test"):
        print("[Preprocessing aircraft data...]")
        get_aircrafts_test_set()

    print("\n")
    print("[Starting Evaluation of Model Checkpoint..]")
    for dataset, total_clients, clients_per_round in zip(datasets, total_clients_in_ds, clients_per_round_in_ds):
        print(f"[Dataset: {dataset}]")
        print("[Starting Evaluation of Model Checkpoint..]")
        num_classes = TABLE_DATASET_CLASSES[dataset]
        client_model = get_transformer_model(
            model_name=model_name,
            classifier_hidden_layers=classifier_hidden_layers,
            num_classes=num_classes,
            random_seed=random_seed,
            load_pretrained_weights=load_pretrained_weights,
            trainable_feature_extractor=trainable_feature_extractor,
            trainable_blocks_fe=trainable_blocks_fe
        )

        # loading model checkpoints from hugging_face hub
        last_checkpoint = 100 if dataset not in ["cifar100"] else 50
        remote_path = "amorale0420/mit-b0_" + dataset + "_R" + str(last_checkpoint) + "_sanitized"
        downloaded_model_sanitized = os.path.join(
            snapshot_download(remote_path),
            "server_model")
        remote_path = "amorale0420/mit-b0_" + dataset + "_R" + str(
            last_checkpoint)
        downloaded_model = os.path.join(
            snapshot_download(remote_path),
            "server_model")

        test_ds = return_test_ds(dataset)

        train_ds_for_test = load_client_datasets_from_files(
            dataset=dataset,
            sampled_client=int(cid),
            total_clients=total_clients,
            batch_size=batch_size,
            alpha=alpha_dirichlet,
            seed=random_seed,
            is_training=False,
        )

        print("[Evaluation] Model trained on Sanitized Dataset (Excluding Client 0)")
        client_model.load_weights(downloaded_model_sanitized)
        client_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        print("--- Test data ---")
        client_model.evaluate(test_ds)
        print("--- Client 0 train data ---")
        client_model.evaluate(train_ds_for_test)

        print("[Evaluation] Model Trained Including Client 0")
        client_model.load_weights(downloaded_model)
        client_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        print("--- Test data ---")
        client_model.evaluate(test_ds)
        print("--- Client 0 train data ---")
        client_model.evaluate(train_ds_for_test)


if __name__ == "__main__":
    main()
