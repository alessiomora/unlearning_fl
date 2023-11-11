"""Client unlearn routine
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any {'0', '1', '2'}

import tensorflow as tf
import tensorflow_datasets as tfds

from unlearning_fl.dataset import (
    get_normalization_fn,
    preprocess_dataset_for_birds_aircafts_cars,
    load_client_datasets_from_files,
    preprocess_dataset_for_transformers_models,
)
from unlearning_fl.model_utility import get_transformer_model


TEST_AIRCRAFTS_DATASET = "/home/amora/pycharm_projects/fed_vit_non_iid/aircrafts_test/test"
TEST_BATCH_SIZE = 128

table_dataset_classes = {
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
    algorithm = "FedAvg"  # can be "FedAvg" or "FedAvg+KD" or "FedMLB"
    model_name = "mit-b0"  # mit-b0, mit-b1, mit-b2, deit_tiny, deit_small
    optimizer_name = "adamw"
    classifier_hidden_layers = 0
    classifier_unit_pl = 0
    load_pretrained_weights = True
    trainable_feature_extractor = True
    trainable_blocks_fe = None
    batch_size = 32
    local_epochs = 5  # number of local epochs
    lr_client = 3e-4  # client learning rate
    exp_decay = 0.998
    clipnorm = None
    l2_weight_decay = 1e-3
    random_seed = 25
    cid = 0

    for dataset, total_clients, clients_per_round in zip(datasets, total_clients_in_ds, clients_per_round_in_ds):
        print(f"[Dataset: {dataset}]")
        print("[Starting Evaluation of Model Checkpoint..]")
        num_classes = table_dataset_classes[dataset]
        client_model = get_transformer_model(
            model_name=model_name,
            classifier_hidden_layers=classifier_hidden_layers,
            num_classes=num_classes,
            random_seed=random_seed,
            load_pretrained_weights=load_pretrained_weights,
            trainable_feature_extractor=trainable_feature_extractor,
            trainable_blocks_fe=trainable_blocks_fe
        )

        # set up the path for the configuration
        # needed to recover the right model checkpoint
        if alpha_dirichlet < 0:
            alpha_dirichlet_string = "iid"
        else:
            alpha_dirichlet_string = "dir_" + str(round(alpha_dirichlet, 2))
        local_batch_size_or_k_defined = "batch_size_" + str(batch_size)
        if clipnorm is None:
            clipnorm_string = "clipnorm_None"
        else:
            clipnorm_string = "clipnorm_" + str(round(clipnorm, 2))

        if not load_pretrained_weights:
            model_name_string = model_name + "_fs"
        else:
            model_name_string = model_name

        if classifier_hidden_layers == 0:
            head_string = "no_hidden_layers"
        else:
            head_string = "chl_" + str(classifier_hidden_layers) + "_" + str(
                classifier_unit_pl)

        feature_extractor_string = "fe_" + str(trainable_feature_extractor)
        trainable_blocks_string = "" if trainable_blocks_fe is None else "_tb_" + str(
            trainable_blocks_fe)

        save_path_checkpoints = os.path.join(
            "model_checkpoints",
            dataset,
            model_name_string,
            algorithm,
            local_batch_size_or_k_defined + "_ep_" + str(local_epochs),
            str(total_clients) + "C_" + str(clients_per_round) + "K",
            alpha_dirichlet_string,
            feature_extractor_string + "_" + trainable_blocks_string,
            optimizer_name + "_lr_client_" + str(round(lr_client, 6)) + "_wd_" + str(
                round(l2_weight_decay, 4)),
            "exp_decay_" + str(round(exp_decay, 3)),
            clipnorm_string,
            head_string,
            "seed_" + str(random_seed),
        )

        # path = os.path.join(save_path_checkpoints, "dict_info.pickle")
        last_checkpoint = 100 if dataset not in ["cifar100"] else 50

        path = os.path.join(
            save_path_checkpoints,
            "checkpoints_R" + str(last_checkpoint),
            "server_model",
        )

        save_path_checkpoints = os.path.join(
            "model_checkpoints",
            dataset + "_sanitized",
            model_name_string,
            algorithm,
            local_batch_size_or_k_defined + "_ep_" + str(local_epochs),
            str(total_clients) + "C_" + str(clients_per_round) + "K",
            alpha_dirichlet_string,
            feature_extractor_string + "_" + trainable_blocks_string,
            optimizer_name + "_lr_client_" + str(round(lr_client, 6)) + "_wd_" + str(
                round(l2_weight_decay, 4)),
            "exp_decay_" + str(round(exp_decay, 3)),
            clipnorm_string,
            head_string,
            "seed_" + str(random_seed),
        )

        path_sanitized = os.path.join(
            save_path_checkpoints,
            "checkpoints_R" + str(last_checkpoint),
            "server_model",
        )

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
        client_model.load_weights(path_sanitized)
        client_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        print("--- Test data ---")
        client_model.evaluate(test_ds)
        print("--- Client 0 train data ---")
        client_model.evaluate(train_ds_for_test)

        print("[Evaluation] Model Trained Including Client 0")
        client_model.load_weights(path)
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
