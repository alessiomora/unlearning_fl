""" This script partions the CIFAR10/CIFAR100 dataset in a federated fashion.
The level of non-iidness is defined via the alpha parameter (alpha in the paper below as well)
for a dirichlet distribution, and rules the distribution of examples per label on clients.
This implementation is based on the paper: https://arxiv.org/abs/1909.06335
"""
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import numpy as np
import os
import shutil
from prepare_aircraft_dataset import (
    fgvc_aircraft_compute_indexes_of_labels,
    read_images_from_filename
)
from prepare_stanford_cars_dataset import (
    load_stanford_cars_dataset
)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def generate_dirichlet_samples(num_of_classes, alpha, num_of_clients,
                               num_of_examples_per_label):
    """Generate samples from a dirichlet distribution based on alpha parameter.
    Samples will have the shape (num_of_clients, num_of_classes).
    Returns an int tensor with shape (num_of_clients, num_of_classes)."""
    for _ in range(0, 10):
        alpha_tensor = tf.fill(num_of_clients, alpha)
        # alpha_tensor = alpha * prior_distrib
        print(alpha_tensor)
        dist = tfp.distributions.Dirichlet(tf.cast(alpha_tensor, tf.float32))
        samples = dist.sample(num_of_classes)
        print(samples)
        # Cast to integer for an integer number of examples per label per client
        int_samples = tf.cast(tf.round(samples * num_of_examples_per_label), tf.int32)
        # int_samples = tf.cast(tf.math.ceil(samples * num_of_examples_per_label), tf.int32)
        int_samples_transpose = tf.transpose(int_samples, [1, 0])
        # print("reduce_sum", tf.reduce_sum(int_samples_transpose, axis=1))
        correctly_generated = tf.reduce_min(tf.reduce_sum(int_samples_transpose, axis=1))
        if tf.cast(correctly_generated, tf.float32) != tf.constant(0.0, tf.float32):
            break
        print("Generated some clients without any examples. Retrying..")

    return int_samples_transpose


def remove_list_from_list(orig_list, to_remove):
    """Remove to_remove list from the orig_list and returns a new list."""
    new_list = []
    for element in orig_list:
        if element not in to_remove:
            new_list.append(element)
    return new_list


def save_dic_as_txt(filename, dic):
    with open(filename, 'w') as file:
        file.write(json.dumps(dic))


if __name__ == '__main__':
    no_repetition = True
    alphas = [10000, 0.1]  # alpha >= 100.0 generates a homogeneous distrib.
    datasets = ["cifar100"]  # dataset = ["cifar100", "birds", "cars", "aircrafts"]
    nums_of_clients = [500]
    table_dataset_classes = {"cifar100": 100, "birds": 200, "cars": 196, "aircrafts": 100}
    table_num_of_examples_per_label = {"cifar100": 500, "birds": 32, "cars": 41,
                                       "aircrafts": 70}
    # total example train, cars: 8,144, birds: 5,994, aircrafts: 6,667
    recover_dataset_name = {"birds": "caltech_birds2011"}
    print("Generating dirichlet partitions..")

    for dataset in datasets:
        for alpha in alphas:
            for num_of_clients in nums_of_clients:
                print(f"Generating alpha = {alpha} partitions for {dataset} with {num_of_clients} sites.")

                client_data_dict = {}
                # preparing folder
                folder = os.path.join(
                    "federated_datasets",
                    dataset + "_dirichlet")
                exist = os.path.exists(folder)

                if not exist:
                    os.makedirs(folder)

                alpha_string = str(round(alpha, 2)) if alpha < 1000 else "iid"
                folder_path = os.path.join(
                    folder,
                    str(num_of_clients),
                    alpha_string)
                exist = os.path.exists(folder_path)

                if not exist:
                    os.makedirs(folder_path)
                else:
                    shutil.rmtree(folder_path, ignore_errors=True)

                num_of_examples_per_label = table_num_of_examples_per_label[dataset]
                num_of_classes = table_dataset_classes[dataset]
                smpls = generate_dirichlet_samples(num_of_classes=num_of_classes, alpha=alpha,
                                                   num_of_clients=num_of_clients,
                                                   num_of_examples_per_label=num_of_examples_per_label)

                if dataset in ["cifar100"]:
                    (x_train, y_train), (_, _) = tf.keras.datasets.cifar100.load_data()
                elif dataset in ["birds"]:
                    dataset_name = recover_dataset_name[dataset]
                    train_ds = tfds.load(dataset_name, split='train',
                                         shuffle_files=False, as_supervised=True)

                    y_train = train_ds.map(lambda x, y: y)
                    y_train_as_list = list(y_train)
                    y_train_as_list_of_np = [t.numpy() for t in y_train_as_list]
                    y_train = np.array(y_train_as_list_of_np)

                    x_train = train_ds.map(lambda x, y: x)
                    x_train_as_list = list(x_train)
                    x_train_as_numpy = [tf.image.resize(t, size=(224, 224)).numpy() for t in x_train_as_list]
                    x_train = np.array(x_train_as_numpy)
                elif dataset in ["aircrafts"]:
                    indexes_of_labels, img_names, labels_associated = fgvc_aircraft_compute_indexes_of_labels()
                elif dataset in ["cars"]:
                    train_ds = load_stanford_cars_dataset(split="train")

                    y_train = train_ds.map(lambda x, y: y)
                    y_train_as_list = list(y_train)
                    y_train_as_list_of_np = [t.numpy() for t in y_train_as_list]
                    y_train = np.array(y_train_as_list_of_np)

                    x_train = train_ds.map(lambda x, y: x)
                    x_train_as_list = list(x_train)
                    x_train_as_numpy = [tf.cast(t, tf.uint8).numpy() for t in x_train_as_list]
                    x_train = np.array(x_train_as_numpy)

                if dataset not in ["aircrafts"]:
                    indexes_of_labels = list([list([]) for _ in range(0, num_of_classes)])

                    j = 0
                    for label in y_train:
                        indexes_of_labels[label.item()].append(j)
                        j = j + 1

                c = 0
                indexes_of_labels_backup = [element for element in indexes_of_labels]
                smpls = smpls.numpy()
                for per_client_sample in smpls:
                    print(f"[Client {c}] Generating dataset..")
                    label = 0

                    list_extracted_all_labels = []

                    for num_of_examples_per_label in per_client_sample:
                        if no_repetition:
                            if len(indexes_of_labels[label]) < num_of_examples_per_label:
                                print(f"label {label} ended")
                                extracted = np.random.choice(indexes_of_labels[label], len(indexes_of_labels[label]), replace=False)
                                smpls[c, label] = smpls[c, label] - len(indexes_of_labels[label])
                            else:
                                extracted = np.random.choice(indexes_of_labels[label], num_of_examples_per_label, replace=False)
                        else:
                            if len(indexes_of_labels[label]) < num_of_examples_per_label:
                                print("[WARNING] Repeated examples!")
                                remained = len(indexes_of_labels[label])
                                extracted_1 = np.random.choice(indexes_of_labels[label], remained, replace=False)
                                indexes_of_labels[label] = indexes_of_labels_backup[label]
                                extracted_2 = np.random.choice(indexes_of_labels[label], num_of_examples_per_label - remained,
                                                               replace=False)
                                extracted = np.concatenate((extracted_1, extracted_2), axis=0)
                            else:
                                extracted = np.random.choice(indexes_of_labels[label], num_of_examples_per_label, replace=False)

                        indexes_of_labels[label] = remove_list_from_list(indexes_of_labels[label], extracted.tolist())

                        for ee in extracted.tolist():
                            list_extracted_all_labels.append(ee)

                        label = label + 1

                    list_extracted_all_labels = list(map(int, list_extracted_all_labels))

                    if dataset not in ["aircrafts"]:
                        numpy_dataset_y = tf.convert_to_tensor(
                            np.asarray(y_train[list_extracted_all_labels]),
                            dtype=tf.int64)
                        # print(type(numpy_dataset_y))
                        numpy_dataset_x = tf.convert_to_tensor(
                            np.asarray(x_train[list_extracted_all_labels]),
                            dtype=tf.uint8)
                        # print(type(numpy_dataset_x))
                        ds = tf.data.Dataset.from_tensor_slices((numpy_dataset_x, numpy_dataset_y))
                        ds = ds.shuffle(buffer_size=4096)

                        tf.data.Dataset.save(ds,
                                                  path=os.path.join(os.path.join(folder_path, "train"),
                                                                    str(c)))
                    else:
                        # img_to_include_in_ds_list = img_names[list_extracted_all_labels]
                        img_to_include_in_ds_list = [img_names[i] for i in list_extracted_all_labels]
                        # y_train = labels_associated[list_extracted_all_labels]
                        y_train = [labels_associated[i] for i in list_extracted_all_labels]

                        images = read_images_from_filename(img_to_include_in_ds_list)
                        numpy_dataset_y = tf.convert_to_tensor(
                            np.asarray(y_train),
                            dtype=tf.int64)

                        ds = tf.data.Dataset.from_tensor_slices((images, numpy_dataset_y))
                        ds = ds.shuffle(buffer_size=4096)

                        tf.data.Dataset.save(ds,
                                                  path=os.path.join(os.path.join(folder_path, "train"),
                                                                    str(c)))

                    # saving the list of image indexes in a dictionary for reproducibility
                    client_data_dict[c] = list_extracted_all_labels
                    c = c + 1

                path = os.path.join(folder_path, "distribution_train.npy")
                np.save(path, smpls)
                smpls_loaded = np.load(path)
                tf.print(smpls_loaded, summarize=-1)
                print("Reduce sum axis label", tf.reduce_sum(smpls_loaded, axis=1))
                print("Reduce sum axis client", tf.reduce_sum(smpls_loaded, axis=0))
                print("Reduce sum ", tf.reduce_sum(smpls_loaded))

                folder_path = os.path.join(
                    "client_data",
                    dataset,
                    "unbalanced",
                )
                if alpha < 1000:
                    file_path = os.path.join(
                        folder_path,
                        "dirichlet"+str(round(alpha, 3))+"_clients"+str(num_of_clients)+".txt"
                    )
                else:
                    file_path = os.path.join(
                        folder_path,
                        "iid" + "_clients" + str(
                            num_of_clients) + ".txt"
                    )
                exist = os.path.exists(folder_path)
                if not exist:
                    os.makedirs(folder_path)

                exist = os.path.exists(file_path)
                if exist:
                    os.remove(file_path)

                save_dic_as_txt(file_path, client_data_dict)
