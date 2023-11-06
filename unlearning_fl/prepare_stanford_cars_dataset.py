import os
import opendatasets as od
import tensorflow as tf

DATA_DIR = './stanford-car-dataset-by-classes-folder/car_data/car_data'


def load_stanford_cars_dataset(split="train"):
    """ Load train or test split as TF dataset instance. """
    if split in ["train", "test"]:
        ds_dir = os.path.join(DATA_DIR, split)

        loaded_ds = tf.keras.preprocessing.image_dataset_from_directory(
            ds_dir,
            image_size=(256, 256),
            batch_size=None,
            label_mode='int',
            shuffle=False,
        )
        return loaded_ds


def compute_mean_and_std(train_ds):
    """Compute the mean and standard dev of the passed TF dataset."""
    means = []
    variances = []
    cont = 0
    for img, _ in train_ds.as_numpy_iterator():
        print(cont)
        img = tf.image.resize(img, size=(224, 224))
        # img = tf.cast(img, tf.uint8)
        mean = tf.math.reduce_mean(img / 255.0, axis=[0, 1])
        variance = tf.math.reduce_variance(img / 255.0, axis=[0, 1])
        means.append(mean)
        variances.append(variance)
        cont = cont +1

    stacked_means = tf.stack(means, axis=0)
    print("Shape ", tf.shape(stacked_means))
    mean = tf.math.reduce_mean(stacked_means, axis=0)
    stacked_var = tf.stack(variances, axis=0)
    std = tf.math.sqrt(tf.reduce_mean(stacked_var, axis=0))
    print("Shape ", tf.shape(stacked_var))

    print("Mean: ", mean, " std: ", std)
    return mean, std


if __name__ == '__main__':

    # dataset_url = 'https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder'
    # od.download(dataset_url)
    train_ds = load_stanford_cars_dataset(split="train")
    compute_mean_and_std(train_ds)
    # Mean: tf.Tensor([0.46683374 0.4599333  0.4504589], shape=(3,), dtype=float32)
    # Std: tf.Tensor([0.26419637 0.2636279  0.26865953], shape=(3,), dtype=float32)
    y_train = train_ds.map(lambda x, y: y)
    y_train_as_list = list(y_train)
    print("Train len: ", len(y_train_as_list))
    occurrence_per_class = tf.math.bincount(tf.stack(tf.cast(y_train_as_list, tf.int32)))
    print("Occurrence per class: ", occurrence_per_class)
    test_ds = load_stanford_cars_dataset(split="test")

    y_test = test_ds.map(lambda x, y: y)
    y_test_as_list = list(y_test)
    print("Test len: ", len(y_test_as_list))
