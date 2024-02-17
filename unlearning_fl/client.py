"""Define TensorFlow client class by subclassing `flwr.client.NumPyClient`."""

from typing import Dict, Tuple

import flwr as fl
import tensorflow as tf
from flwr.common import NDArrays, Scalar


def write_on_file(to_print_string: str, dataset: str, alpha_dir: float):
    if alpha_dir < 0:
        filename = dataset + "_iid" + "_unlearning_client_participation.txt"
    else:
        filename = dataset + "_dir" + str(round(alpha_dir, 2)) +"_unlearning_client_participation.txt"
    with open(filename, 'a') as file:
        file.write(to_print_string + "\n")


class TFClient(fl.client.NumPyClient):
    """Tensorflow Client implementation."""

    def __init__(
        self,
        train_ds: tf.data.Dataset,
        model: tf.keras.Model,
        num_examples_train: int,
        algorithm: str,
        dataset: str,
        cid: int,
        alpha_dir : float,
    ):
        self.model = model
        self.train_ds = train_ds
        self.num_examples_train = num_examples_train
        self.algorithm = algorithm
        self.cid = cid
        self.dataset = dataset
        self.alpha_dir = alpha_dir

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get the local model parameters."""
        return self.model.get_weights()

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Train parameters on the locally held training set."""
        epochs: int = int(config["local_epochs"])
        current_round: int = int(config["current_round"])
        exp_decay: float = float(config["exp_decay"])
        lr_client_initial: float = float(config["lr_client_initial"])

        print("[CID] ", self.cid)
        if self.cid == 0:
            # write_on_file(f"[Round {current_round}] Client {self.cid} has participated")
            write_on_file(f"{current_round}", dataset=self.dataset, alpha_dir=self.alpha_dir)

        if current_round > 1:
            lr_client = lr_client_initial * (exp_decay ** (current_round - 1))
            # During training, update the learning rate as needed
            tf.keras.backend.set_value(self.model.optimizer.lr, lr_client)

        # Update local model parameters
        if self.algorithm in ["FedMLB", "FedAvg+KD"]:
            self.model.local_model.set_weights(parameters)
            self.model.global_model.set_weights(parameters)
        elif self.algorithm in ["FedAvg"]:
            self.model.set_weights(parameters)
        else:  # fedsmoothie
            self.model.local_model.set_weights(parameters)

        # self.model.summary()
        # Get hyperparameters for this round
        # batch_size: int = config["batch_size"]
        # the dataset is already batched,
        # so there is no need to retrieve the batch size

        # in model.fit it is not mandatory to specify
        # batch_size if the dataset is already batched
        # as in our case
        results = self.model.fit(self.train_ds, epochs=epochs, verbose=0)
        # results = self.model.fit(self.train_ds, epochs=epochs)
        # results = self.model.fit(self.train_ds, epochs=epochs, validation_data=test_ds)

        # test_aircrafts_path = "./aircrafts_test/test"
        # test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        #     test_aircrafts_path,
        #     image_size=(256, 256),
        #     batch_size=None,
        #     label_mode='int',
        # )
        # test_ds = (
        #     test_ds  # .map(cut_lower_20px_fn)
        #         .map(preprocess_dataset_for_birds_and_aircafts(is_training=False))
        #         .map(get_normalization_fn("mit-b0", dataset="aircrafts"))
        #         .batch(32)
        # )
        # self.model.evaluate(test_ds)
        # results = self.model.fit(self.train_ds, epochs=epochs, validation_data=test_ds)

        parameters_prime = self.model.get_weights()
        num_examples_train = self.num_examples_train

        return parameters_prime, int(num_examples_train), results.history
