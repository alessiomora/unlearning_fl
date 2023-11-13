"""Assessing the performance of global models and sanitized global models
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any {'0', '1', '2'}

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


TEST_AIRCRAFTS_DATASET = "/home/amora/pycharm_projects/fed_vit_non_iid/aircrafts_test/test"
TEST_BATCH_SIZE = 128

TABLE_DATASET_CLASSES = {
    "cifar100": 100,
    "birds": 200,
    "cars": 196,
    "aircrafts": 100
}


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
    print("[Starting Evaluation of Model Checkpoint..]")
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
