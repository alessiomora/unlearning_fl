"""Client unlearn routine
"""
import os

from os import walk

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any {'0', '1', '2'}


import os
import tensorflow as tf
import tensorflow_datasets as tfds
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from unlearning_fl.dataset import (
    get_normalization_fn,
    preprocess_dataset_for_birds_aircafts_cars,
    load_client_datasets_from_files,
)
from unlearning_fl.model_utility import get_transformer_model
from unlearning_fl.fedsmoothie_model import FedSmoothieModel
from unlearning_fl.fedmixup_model import FedMixUpModel
from unlearning_fl.goodbad_model import GoodBadModel
from unlearning_fl.mixup_utility import (
    return_augmented_dataset,
    return_bad_good_dataset,
)
from unlearning_fl.utils import (
    dic_load,
)


TEST_BATCH_SIZE = 128
IMAGE_SIZE = 224


def get_optimizer(optimizer_name="adamw", lr_client=1e-5, clipnorm=None, l2_weight_decay=1e-3, momentum=0.0):
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_client,
            clipnorm=clipnorm,
            weight_decay=l2_weight_decay,
        )

    elif optimizer_name == "adamw":
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr_client,
            clipnorm=clipnorm,
            weight_decay=l2_weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr_client,
            clipnorm=clipnorm,
            weight_decay=l2_weight_decay,
        )
    else:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr_client,
            clipnorm=clipnorm,
            weight_decay=l2_weight_decay,
            momentum=momentum,
        )
    return optimizer


table_dataset_classes = {"cifar100": 100, "birds": 200, "cars": 196,
                             "aircrafts": 100}


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:  # pylint: disable=too-many-locals
    print("[Start Simulation]")
    # Print parsed config
    print(OmegaConf.to_yaml(cfg))
    cid = 0  # unlearning client's id
    load_pretrained_weights = cfg.load_pretrained_weights
    trainable_feature_extractor = cfg.trainable_feature_extractor
    model_name = cfg.model_name
    momentum = cfg.momentum
    optimizer_name = cfg.optimizer
    # two_layer_classifier = cfg.two_layer_classifier
    classifier_hidden_layers = cfg.classifier_hidden_layers
    classifier_unit_pl = cfg.classifier_unit_pl
    algorithm = cfg.algorithm
    random_seed = cfg.random_seed
    lr_client = cfg.lr_client
    # exp_decay = cfg.exp_decay
    clipnorm = cfg.clipnorm
    l2_weight_decay = cfg.l2_weight_decay
    alpha_dirichlet = cfg.dataset_config.alpha_dirichlet
    # local_updates = cfg.local_updates
    local_epochs = cfg.local_epochs
    total_clients = cfg.total_clients
    dataset = cfg.dataset_config.dataset
    # restart_from_checkpoint = cfg.restart_from_checkpoint
    batch_size = cfg.batch_size
    trainable_blocks_fe = cfg.trainable_blocks
    num_classes = table_dataset_classes[dataset]
    configs = []
    # configs.append([1, 5e-3])  # [epochs, smoothing]
    # configs.append([1, 1e-3])  # [epochs, smoothing]
    # configs.append([1, 5e-4])  # [epochs, smoothing]
    # configs.append([1, 1e-4])  # [epochs, smoothing]
    # configs.append([1, 9e-4])  # [epochs, smoothing]
    # configs.append([1, 7e-4])  # [epochs, smoothing]
    # configs.append([1, 5e-4])  # [epochs, smoothing]
    # configs.append([1, 3e-4])  # [epochs, smoothing]
    # configs.append([1, 1e-4])  # [epochs, smoothing]
    # configs.append([1, 9e-5])  # [epochs, smoothing]
    # configs.append([1, 7e-5])  # [epochs, smoothing]
    # configs.append([1, 5e-5])  # [epochs, smoothing]
    # configs.append([1, 4e-5])  # [epochs, smoothing]
    configs.append([1, 3e-5])  # [epochs, smoothing]
    configs.append([1, 1e-5])  # [epochs, smoothing]
    configs.append([1, 9e-6])  # [epochs, smoothing]
    configs.append([1, 8e-6])  # [epochs, smoothing]
    configs.append([1, 7e-6])  # [epochs, smoothing]
    # configs.append([1, 9e-6])  # [epochs, smoothing]
    # configs.append([1, 8e-6])  # [epochs, smoothing]
    # configs.append([1, 6e-6])  # [epochs, smoothing]

    # configs.append([1, 1e-7])
    # configs.append([2, 1e-6])  # [epochs, smoothing]
    # configs.append([2, 3e-6])
    # configs.append([2, 6e-6])
    # configs.append([2, 1e-7])
    # configs.append([1, 0.5])
    # configs.append([1, 0.7])
    # configs.append([1, 0.8])
    # configs.append([2, 0.1])
    # configs.append([2, 0.2])
    # configs.append([2, 0.3])
    # configs.append([5, 0.1])
    # configs.append([5, 0.2])
    # configs.append([5, 0.3])
    # configs.append([5, 0.5])
    # configs.append([5, 0.7])
    # configs.append([5, 0.8])

    client_model = get_transformer_model(
        model_name=model_name,
        classifier_hidden_layers=classifier_hidden_layers,
        num_classes=num_classes,
        random_seed=random_seed,
        load_pretrained_weights=load_pretrained_weights,
        trainable_feature_extractor=trainable_feature_extractor,
        trainable_blocks_fe=trainable_blocks_fe
    )
    random_model = get_transformer_model(
        model_name=model_name,
        classifier_hidden_layers=classifier_hidden_layers,
        num_classes=num_classes,
        random_seed=random_seed,
        load_pretrained_weights=load_pretrained_weights,
        trainable_feature_extractor=trainable_feature_extractor,
        trainable_blocks_fe=trainable_blocks_fe
    )

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
        str(total_clients) + "C_" + str(cfg.clients_per_round)+"K",
        alpha_dirichlet_string,
        feature_extractor_string + "_" + trainable_blocks_string,
        optimizer_name + "_lr_client_" + str(round(lr_client, 6)) + "_wd_" + str(
            round(l2_weight_decay, 4)),
        "exp_decay_" + str(round(cfg.exp_decay, 3)),
        clipnorm_string,
        head_string,
        "seed_" + str(random_seed),
    )

    # path = os.path.join(save_path_checkpoints, "dict_info.pickle")
    last_checkpoint = 100

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
        str(total_clients) + "C_" + str(cfg.clients_per_round)+"K",
        alpha_dirichlet_string,
        feature_extractor_string + "_" + trainable_blocks_string,
        optimizer_name + "_lr_client_" + str(round(lr_client, 6)) + "_wd_" + str(
            round(l2_weight_decay, 4)),
        "exp_decay_" + str(round(cfg.exp_decay, 3)),
        clipnorm_string,
        head_string,
        "seed_" + str(random_seed),
    )

    path_sanitized = os.path.join(
            save_path_checkpoints,
            "checkpoints_R" + str(last_checkpoint),
            "server_model",
    )

    test_ds = tfds.load("caltech_birds2011", split='test',
                        shuffle_files=False, as_supervised=True)
    test_ds = (
        test_ds.map(
            preprocess_dataset_for_birds_aircafts_cars(is_training=False))
            .map(get_normalization_fn(model_name, dataset="birds"))
            .batch(TEST_BATCH_SIZE)
    )

    train_ds_for_test = load_client_datasets_from_files(
        dataset=dataset,
        sampled_client=int(cid),
        total_clients=total_clients,
        batch_size=batch_size,
        alpha=alpha_dirichlet,
        seed=random_seed,
        is_training=False,
    )

    train_ds_for_train = load_client_datasets_from_files(
        dataset=dataset,
        sampled_client=int(cid),
        total_clients=total_clients,
        batch_size=batch_size,
        alpha=alpha_dirichlet,
        seed=random_seed,
        is_training=True,  # enabling random augmentations
    )
    train_ds_other_clients = load_client_datasets_from_files(
        dataset=dataset,
        sampled_client=1,
        total_clients=total_clients,
        batch_size=batch_size,
        alpha=alpha_dirichlet,
        seed=random_seed,
        is_training=True,  # enabling random augmentations
    )
    for i in range(2, 29):
        train_ds_for_train_client_2 = load_client_datasets_from_files(
            dataset=dataset,
            sampled_client=i,
            total_clients=total_clients,
            batch_size=batch_size,
            alpha=alpha_dirichlet,
            seed=random_seed,
            is_training=True,  # enabling random augmentations
        )
        train_ds_other_clients = train_ds_other_clients.concatenate(train_ds_for_train_client_2)

    # train_ds_other_clients = train_ds_for_train_client_1.concatenate(train_ds_for_train_client_2)
    train_ds_other_clients = train_ds_other_clients.unbatch().shuffle(1024).batch(32)


    print("[Evaluation] Model trained on Sanitized Dataset (Excluding Client 0)")
    client_model.load_weights(path_sanitized)
    client_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    # print("--- Test data ---")
    # client_model.evaluate(test_ds)
    # print("--- Client 0 train data ---")
    # client_model.evaluate(train_ds_for_test)

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

    for cfg in configs:
        print("\n")
        print("------- Unlearning -------")
        client_model.load_weights(path)
        epochs = cfg[0]
        lr = cfg[1]
        print(f"[Config] epochs {epochs} lr {lr}")
        optimizer = get_optimizer(lr_client=lr)
        # model = FedSmoothieModel(client_model)
        # model.compile(
        #     optimizer=optimizer,
        #     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        #     beta=smoothing,
        # )
        # results = model.fit(train_ds_for_train, epochs=epochs)
        # model.compile(
        #     optimizer=optimizer,
        #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        # )
        # print("[Test data]")
        # model.evaluate(test_ds)
        # print("[Client 0 train data]")
        # model.evaluate(train_ds_for_test)

        # model = FedMixUpModel(client_model)
        model = GoodBadModel(client_model, random_model)
        # client_model.classifier.set_weights(random_model.classifier.get_weights())
        # model.compile(
        #     optimizer=optimizer,
        #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        # )
        # print("[Test data]")
        # model.evaluate(test_ds)
        # print("[Client 0 train data]")
        # model.evaluate(train_ds_for_test)
        model.compile(
            optimizer=optimizer,
            # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            loss=tf.keras.losses.KLDivergence(),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
        )

        # train_ds_mix_up = return_augmented_dataset(train_ds_for_train, num_classes=num_classes)
        # train_ds_mix_up = train_ds_mix_up.unbatch().shuffle(1024).batch(32)

        train_ds_mix_up = return_bad_good_dataset(bad_knowledge=train_ds_for_train, good_knowledge=train_ds_other_clients)
        train_ds_mix_up = train_ds_mix_up.unbatch().shuffle(1024*24).batch(32)
        model.fit(train_ds_mix_up, epochs=epochs)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        print("[Test data]")
        model.evaluate(test_ds)
        print("[Client 0 train data]")
        model.evaluate(train_ds_for_test)

        # client_model.compile(
        #     optimizer=optimizer,
        #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        # )
        # client_model.fit(train_ds_other_clients, epochs=3)
        # print("[Test data]")
        # client_model.evaluate(test_ds)
        # print("[Client 0 train data]")
        # client_model.evaluate(train_ds_for_test)



if __name__ == "__main__":
    main()


# [Round 8] Client 0 has participated
# [Round 13] Client 0 has participated
# [Round 14] Client 0 has participated
# [Round 16] Client 0 has participated
# [Round 19] Client 0 has participated
# [Round 21] Client 0 has participated
# [Round 22] Client 0 has participated
# [Round 27] Client 0 has participated
# [Round 37] Client 0 has participated
# [Round 38] Client 0 has participated
# [Round 39] Client 0 has participated
# [Round 42] Client 0 has participated
# [Round 45] Client 0 has participated
# [Round 47] Client 0 has participated
# [Round 59] Client 0 has participated
# [Round 61] Client 0 has participated
# [Round 63] Client 0 has participated
# [Round 64] Client 0 has participated
# [Round 65] Client 0 has participated
# [Round 68] Client 0 has participated
# [Round 77] Client 0 has participated
# [Round 86] Client 0 has participated
# [Round 92] Client 0 has participated