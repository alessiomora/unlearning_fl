"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import os
import shutil
from typing import Callable, Dict, Optional, Tuple

# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import flwr
import hydra
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from flwr.common import NDArrays, Scalar
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from unlearning_fl.dataset import (
    get_normalization_fn,
    preprocess_dataset_for_transformers_models,
    preprocess_dataset_for_birds_aircafts_cars,
    load_selected_client_statistics,
    load_client_datasets_from_files,
)
from unlearning_fl.prepare_stanford_cars_dataset import (
    load_stanford_cars_dataset
)
import unlearning_fl.models as models
from unlearning_fl.client import TFClient
from unlearning_fl.fedavg_kd_model import FedAvgKDModel
from unlearning_fl.fedmlb_model import FedMLBModel
from unlearning_fl.model_utility import get_transformer_model
from unlearning_fl.models import create_resnet18
from unlearning_fl.server import MyServer
from unlearning_fl.utils import (
    dic_load,
    dic_save,
    get_cpu_memory,
    get_gpu_memory,
    save_results_as_pickle,
)

# Make TensorFlow logs less verbose


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
enable_tf_gpu_growth()

TEST_BATCH_SIZE = 256

test_aircrafts_path = "./aircrafts_test/test"


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:  # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    print("[Start Simulation]")
    # Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # def get_normalization_fn(model_name="mit-b0"):
    #     if model_name.startswith("mit"):
    #         transpose = True
    #     else:
    #         transpose = False
    #
    #     def element_norm_cifar100_fn(image, label):
    #         """Normalize cifar100 images."""
    #         norm_layer = tf.keras.layers.Normalization(
    #             mean=[0.5071, 0.4865, 0.4409],
    #             variance=[np.square(0.2673), np.square(0.2564), np.square(0.2762)],
    #         )
    #         if transpose:
    #             return tf.transpose(norm_layer(tf.cast(image, tf.float32) / 255.0),
    #                                 (2, 0, 1)), label
    #         return norm_layer(tf.cast(image, tf.float32) / 255.0), label
    #
    #     return element_norm_cifar100_fn

    def get_evaluate_fn(
            model_name: str, model: tf.keras.Model, save_path: str, dataset: str,
            starting_round: int
    ) -> Callable[
        [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
    ]:
        """Return an evaluation function for server-side evaluation."""
        if dataset in ["cifar100"]:
            (_, _), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
            test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            if model_name.startswith("resnet"):
                test_ds = test_ds.map(get_normalization_fn(model_name)).batch(
                    TEST_BATCH_SIZE)
            else:  # transformer models
                test_ds = (
                    test_ds.map(
                        preprocess_dataset_for_transformers_models(is_training=False))
                        .map(get_normalization_fn(model_name))
                        .batch(TEST_BATCH_SIZE)
                )
        elif dataset in ["birds"]:
            test_ds = tfds.load("caltech_birds2011", split='test',
                                shuffle_files=False, as_supervised=True)
            test_ds = (
                test_ds.map(
                    preprocess_dataset_for_birds_aircafts_cars(is_training=False))
                    .map(get_normalization_fn(model_name, dataset="birds"))
                    .batch(TEST_BATCH_SIZE)
            )
        elif dataset in ["aircrafts"]:
            # print("------- AIRCRAFTS -----------")
            test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                test_aircrafts_path,
                image_size=(256, 256),
                batch_size=None,
                label_mode='int',
            )
            test_ds = (
                test_ds  # .map(cut_lower_20px_fn)
                    .map(preprocess_dataset_for_birds_aircafts_cars(is_training=False))
                    .map(get_normalization_fn(model_name, dataset="aircrafts"))
                    .batch(TEST_BATCH_SIZE)
            )
        else:
            test_ds = load_stanford_cars_dataset(split="test")
            test_ds = (
                test_ds
                    .map(preprocess_dataset_for_birds_aircafts_cars(is_training=False))
                    .map(get_normalization_fn(model_name, dataset="cars"))
                    .batch(TEST_BATCH_SIZE)
            )

        # creating a tensorboard writer to log results
        # then results can be monitored in real-time with tensorboard
        # running the command:
        # tensorboard --logdir [HERE_THE_PATH_OF_TF_BOARD_LOGS]
        global_summary_writer = tf.summary.create_file_writer(save_path)

        # The `evaluate` function will be called after every round
        def evaluate(
                server_round: int,
                parameters: NDArrays,
                config: Dict[str, Scalar],  # pylint: disable=unused-argument
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            model.set_weights(parameters)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(
                test_ds,
            )

            with global_summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=server_round)
                tf.summary.scalar("accuracy", accuracy, step=server_round)

                if cfg.logging_memory_usage:
                    # logging metrics on memory usage
                    gpu_free_memory = get_gpu_memory()
                    cpu_free_memory = get_cpu_memory()
                    tf.summary.scalar(
                        "cpu_free_mem", cpu_free_memory, step=server_round
                    )
                    tf.summary.scalar(
                        "gpu_free_mem", gpu_free_memory, step=server_round
                    )

            # saving the checkpoint before the end of simulation
            if cfg.save_checkpoint and server_round == (
                    cfg.num_rounds + starting_round - 1
            ):
                path = os.path.join(
                    save_path_checkpoints,
                    "checkpoints_R" + str(server_round),
                    "server_model",
                )
                server_model.save_weights(path)

                path = os.path.join(save_path_checkpoints, "dict_info")
                dic_save({"checkpoint_round": server_round}, path)

            return loss, {"accuracy": accuracy}

        return evaluate

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return training configuration dict for each round."""
        config = {
            "current_round": server_round,
            "local_epochs": cfg.local_epochs,
            "exp_decay": cfg.exp_decay,
            "lr_client_initial": cfg.lr_client,
        }
        return config

    ray_init_args = {"include_dashboard": False}
    # Parse input parameters
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
    local_updates = cfg.local_updates
    local_epochs = cfg.local_epochs
    total_clients = cfg.total_clients
    dataset = cfg.dataset_config.dataset
    restart_from_checkpoint = cfg.restart_from_checkpoint
    batch_size = cfg.batch_size
    trainable_blocks_fe = cfg.trainable_blocks

    table_dataset_classes = {"cifar100": 100, "birds": 200, "cars": 196,
                             "aircrafts": 100}

    # if cfg.batch_size is set to null,
    # local_batch_size = round(local_examples * local_epochs / local_updates)
    # if cfg.batch_size is set to a value, it will be used as local_batch_size
    if batch_size is None:
        local_batch_size_or_k_defined = "K_" + str(local_updates)
    else:
        local_batch_size_or_k_defined = "batch_size_" + str(batch_size)

    if dataset in ["cifar100"]:
        # num_classes = 100
        input_shape = (None, 32, 32, 3)
    else:  # tiny-imagenet
        # num_classes = 200
        input_shape = (None, 64, 64, 3)

    num_classes = table_dataset_classes[dataset]

    def client_fn(cid: str) -> TFClient:
        """Instantiate TF Client."""
        local_examples = load_selected_client_statistics(
            int(cid),
            total_clients=total_clients,
            alpha=alpha_dirichlet,
            dataset=dataset,
        )

        # if cfg.batch_size is set to null,
        # local_batch_size = round(local_examples * local_epochs / local_updates)
        # if cfg.batch_size is set to a value, it will be used as local_batch_size
        if batch_size is None:
            local_batch_size = round(local_examples * local_epochs / local_updates)
        else:
            local_batch_size = batch_size

        training_dataset = load_client_datasets_from_files(
            dataset=dataset,
            sampled_client=int(cid),
            total_clients=total_clients,
            batch_size=local_batch_size,
            alpha=alpha_dirichlet,
            seed=random_seed,
        )

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
        if algorithm in ["FedAvg"]:
            if model_name.startswith("resnet"):
                client_model = models.create_resnet18(
                    num_classes=num_classes,
                    input_shape=input_shape,
                    norm="group",
                    seed=random_seed,
                )
            else:
                client_model = get_transformer_model(
                    model_name=model_name,
                    classifier_hidden_layers=classifier_hidden_layers,
                    num_classes=num_classes,
                    random_seed=random_seed,
                    load_pretrained_weights=load_pretrained_weights,
                    trainable_feature_extractor=trainable_feature_extractor,
                    trainable_blocks_fe=trainable_blocks_fe
                )
                # client_model.summary(expand_nested=True, show_trainable=True)

            client_model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                ),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            )

        else:  # algorithm in ["FedAvg+KD"]:
            if model_name.startswith("resnet"):
                local_model = models.create_resnet18(
                    num_classes=num_classes,
                    input_shape=input_shape,
                    norm="group",
                    seed=random_seed,
                )
                global_model = models.create_resnet18(
                    num_classes=num_classes,
                    input_shape=input_shape,
                    norm="group",
                    seed=random_seed,
                )
            else:
                local_model = get_transformer_model(
                    model_name=model_name,
                    classifier_hidden_layers=classifier_hidden_layers,
                    classifier_unit_pl=classifier_unit_pl,
                    num_classes=num_classes,
                    load_pretrained_weights=load_pretrained_weights,
                    trainable_feature_extractor=trainable_feature_extractor,
                    random_seed=random_seed,
                    trainable_blocks_fe=trainable_blocks_fe
                )
                global_model = get_transformer_model(
                    model_name=model_name,
                    classifier_hidden_layers=classifier_hidden_layers,
                    classifier_unit_pl=classifier_unit_pl,
                    num_classes=num_classes,
                    random_seed=random_seed)

            kd_loss = tf.keras.losses.KLDivergence(
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
            )
            client_model = FedAvgKDModel(local_model, global_model, kd_loss, gamma=0.2)

            client_model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                ),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            )

        client = TFClient(training_dataset, client_model, local_examples, algorithm, cid=cid)

        # Create and return client
        return client

    if model_name.startswith("resnet"):
        server_model = create_resnet18(
            num_classes=num_classes,
            input_shape=input_shape,
            norm="group",
            seed=cfg.random_seed,
        )
    else:
        server_model = get_transformer_model(
            model_name=model_name,
            classifier_hidden_layers=classifier_hidden_layers,
            classifier_unit_pl=classifier_unit_pl,
            num_classes=num_classes,
            load_pretrained_weights=load_pretrained_weights,
            trainable_feature_extractor=True,
            trainable_blocks_fe=trainable_blocks_fe,
            random_seed=random_seed)

    server_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    server_model.summary(expand_nested=True)
    if alpha_dirichlet < 0:
        alpha_dirichlet_string = "iid"
    else:
        alpha_dirichlet_string = "dir_" + str(round(alpha_dirichlet, 2))
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

    sanitized_dataset_string = "_sanitized" if cfg.sanitized_dataset else ""
    save_path_logging = os.path.join(
        "logging_results",
        # "logging_hp_search",
        dataset + sanitized_dataset_string,
        model_name_string,
        algorithm,
        local_batch_size_or_k_defined + "_ep_" + str(local_epochs),
        str(total_clients) + "C_" + str(cfg.clients_per_round)+"K",
        alpha_dirichlet_string,
        feature_extractor_string + trainable_blocks_string,
        optimizer_name + "_lr_client_" + str(round(lr_client, 6)) + "_wd_" + str(
            round(l2_weight_decay, 4)),
        "exp_decay_" + str(round(cfg.exp_decay, 3)),
        clipnorm_string,
        head_string,
        "seed_" + str(random_seed),
    )

    save_path_checkpoints = os.path.join(
        "model_checkpoints",
        dataset + sanitized_dataset_string,
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

    starting_round = 1
    if restart_from_checkpoint:
        # if there is a checkpoint and restart_from_checkpoint is True
        # the training restart from the state saved in the most recent checkpoint
        # i.e., the one indicated in a dictionary named dict_info
        path = os.path.join(save_path_checkpoints, "dict_info.pickle")
        last_checkpoint = dic_load(path)["checkpoint_round"]
        if last_checkpoint:
            print(f"Loading saved checkpoint round {last_checkpoint}")
            path = os.path.join(
                save_path_checkpoints,
                "checkpoints_R" + str(last_checkpoint),
                "server_model",
            )
            server_model.load_weights(path)
            starting_round = last_checkpoint + 1
    else:
        # this will delete the checkpoints of previous simulations for that config
        exist = os.path.exists(save_path_checkpoints)
        if exist:
            shutil.rmtree(save_path_checkpoints, ignore_errors=True)

    tf.keras.utils.set_random_seed(cfg.random_seed * starting_round)
    params = server_model.get_weights()

    min_available_clients = (cfg.total_clients - 1) if cfg.sanitized_dataset else cfg.total_clients
    strategy = instantiate(
        cfg.strategy,
        min_available_clients=min_available_clients,
        initial_parameters=flwr.common.ndarrays_to_parameters(params),
        evaluate_fn=get_evaluate_fn(
            model_name, server_model, save_path_logging, dataset, starting_round,
        ),
        on_fit_config_fn=fit_config,
    )

    my_server = MyServer(strategy=strategy, starting_round=starting_round)

    list_of_clients_ids = [idx for idx in range(cfg.total_clients)]
    if cfg.sanitized_dataset:
        # first client_idx corresponds to the unlearned client
        list_of_clients_ids.pop(0)

    # Start Flower simulation
    history = flwr.simulation.start_simulation(
        client_fn=client_fn,
        # num_clients=cfg.total_clients,
        clients_ids=list_of_clients_ids,
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        server=my_server,
        config=flwr.server.ServerConfig(num_rounds=cfg.num_rounds),
        ray_init_args=ray_init_args,
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth
            # Enable GPU growth upon actor init
            # does nothing if `num_gpus` in client_resources is 0.0
        },
    )

    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path)


if __name__ == "__main__":
    main()

# psutil = "5.9.5" # this is just to log some info about memory usage
# ml_collections = "0.1.1"
# tensorflow-hub = "0.14.0"
# nohup python -m fed_vit_non_iid.main --multirun local_epochs=1,5 optimizer="adamw" batch_size=32 dataset_config.alpha_dirichlet=-1 num_rounds=30 trainable_feature_extractor=True lr_client=1e-3,1e-4,3e-4,6e-4,1e-5,1e-6
# nohup python -m fed_vit_non_iid.main --multirun model_name="deit_tiny","deit_small" two_layer_classifier=False,True batch_size=16,32,128
# nohup python -m fed_vit_non_iid.main --multirun model_name="deit_tiny","deit_small" exp_decay=1.0,0.998,0.995 two_layer_classifier=False,True lr_client=0.3,0.1,0.05,0.01,0.001
