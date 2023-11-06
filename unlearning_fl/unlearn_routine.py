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
from unlearning_fl.utils import (
    dic_load,
)

TEST_BATCH_SIZE = 128
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


def get_optimizer(optimizer_name="adamw", lr_client=3e-4, clipnorm=None, l2_weight_decay=1e-3, momentum=0.0):
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
    configs[0] = [1, 0.1]
    configs[1] = [1, 0.2]
    configs[2] = [1, 0.3]
    configs[3] = [1, 0.5]
    configs[4] = [1, 0.7]
    configs[5] = [1, 0.8]
    configs[6] = [5, 0.1]
    configs[7] = [5, 0.2]
    configs[8] = [5, 0.3]
    configs[9] = [5, 0.5]
    configs[10] = [5, 0.7]
    configs[11] = [5, 0.8]



    client_model = get_transformer_model(
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
    last_checkpoint = 50

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

    train_ds = load_client_datasets_from_files(
        dataset=dataset,
        sampled_client=int(cid),
        total_clients=total_clients,
        batch_size=batch_size,
        alpha=alpha_dirichlet,
        seed=random_seed,
    )

    print("[Evaluation] Model trained on Sanitized Dataset")
    client_model.load_weights(path_sanitized)
    client_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    client_model.evaluate(test_ds)

    print("[Evaluation] Model before Unlearning")
    client_model.load_weights(path)
    client_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    client_model.evaluate(test_ds)

    for cfg in configs:
        print("------- Unlearning -------")
        client_model.load_weights(path)
        epochs = cfg[0]
        smoothing = cfg[1]
        print(f"Config: epochs {epochs} smoothing {smoothing}")
        optimizer = get_optimizer()
        model = FedSmoothieModel(client_model)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            beta=smoothing,
        )
        results = model.fit(train_ds, epochs=epochs, verbose=0)
        model.evaluate(test_ds)