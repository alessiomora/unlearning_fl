# wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
# mkdir fgvc-aircraft
# tar -xvzf fgvc-aircraft-2013b.tar.gz -C fgvc-aircraft

import shutil
import os
from PIL import Image
import numpy as np
import tensorflow as tf


def fgvc_aircraft_compute_indexes_of_labels():
    num_of_classes = 100
    data_path = "/home/amora/fgvc-aircraft/fgvc-aircraft-2013b/data"
    variants_label_path = os.path.join(data_path,
                                       "variants.txt")  # contains all the labels
    train_val_path = os.path.join(data_path,
                                  "images_variant_trainval.txt")  # img.jpg label
    # test_path = os.path.join(data_path, "images_variant_test.txt")

    # indexes_of_labels
    # [[],[],...]  label_0 [23, 43, 12]
    convert_label_to_index_dict = {}
    all_variants = []
    variant_label_id = 0
    with open(variants_label_path) as train_file_txt:
        for line in train_file_txt:
            variant_as_string = line.split("\n")[0]  # remove the ending line
            # print(variant_as_string)
            convert_label_to_index_dict[variant_as_string] = variant_label_id
            variant_label_id = variant_label_id + 1
            all_variants.append(variant_as_string)
    myset = set(all_variants)
    print("Total classes ", len(list(myset)))
    # print(convert_label_to_index_dict)
    indexes_of_labels = list([list([]) for _ in range(0, num_of_classes)])
    names_of_images_associated_to_index = []
    label_of_images_associated_to_index = []
    img_count = 0

    with open(train_val_path) as train_file_txt:
        for line in train_file_txt:
            split = line.split(maxsplit=1)
            img_file_name = split[0]
            # label = split[1].split("\n")[0]  # remove the ending line
            # if label in convert_label_to_index_dict:
            #     label_id = convert_label_to_index_dict[label]
            # else:
            #     print("Not present " + label)
            label_id = convert_label_to_index_dict[split[1].split("\n")[0]]

            indexes_of_labels[label_id].append(img_count)
            names_of_images_associated_to_index.append(img_file_name)
            label_of_images_associated_to_index.append(label_id)

            img_count = img_count + 1
    print("Total training images ", img_count)
    return indexes_of_labels, names_of_images_associated_to_index, label_of_images_associated_to_index


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


def read_images_from_filename(list_of_images_names):
    data_path = "/home/amora/fgvc-aircraft/fgvc-aircraft-2013b/data/images"
    images = []
    for img_name in list_of_images_names:
        img_path = os.path.join(data_path, img_name + ".jpg")
        img = pil_loader(img_path)
        img_tf = cut_lower_20px_fn(img)
        img_tf = tf.cast(tf.image.resize(img_tf, size=(256, 256)), tf.uint8)
        images.append(img_tf)
    stacked_images = tf.stack(images)
    return stacked_images


def compute_mean_and_variance():
    data_path_img = "/home/amora/fgvc-aircraft/fgvc-aircraft-2013b/data/images"
    data_path = "/home/amora/fgvc-aircraft/fgvc-aircraft-2013b/data"
    train_val_path = os.path.join(data_path, "images_variant_trainval.txt")

    images = []
    means = []
    variances = []

    with open(train_val_path) as train_file_txt:
        for line in train_file_txt:
            split = line.split(maxsplit=1)
            img_file_name = split[0]
            img_path = os.path.join(data_path_img, img_file_name + ".jpg")
            img = pil_loader(img_path)
            img_np = np.asarray(img)

            target_height = img_np.shape[0] - 20
            target_width = img_np.shape[1]
            img = tf.image.crop_to_bounding_box(
                img_np, offset_height=0, offset_width=0, target_height=target_height,
                target_width=target_width
            )
            # im_to_save = Image.fromarray(img_np.numpy())
            # data_path = "/home/amora/fgvc-aircraft/fgvc-aircraft-2013b/data/"
            # im_to_save.save(os.path.join(data_path, "cropped.png"))
            img = tf.image.resize(img, size=(224, 224))
            print(tf.shape(img))
            mean = tf.math.reduce_mean(img / 255.0, axis=[0, 1])
            print(tf.shape(mean))
            variance = tf.math.reduce_variance(img / 255.0, axis=[0, 1])
            means.append(mean)
            variances.append(variance)

    stacked_means = tf.stack(means, axis=0)
    print("Shape ", tf.shape(stacked_means))
    mean = tf.math.reduce_mean(stacked_means, axis=0)
    stacked_var = tf.stack(variances, axis=0)
    std = np.sqrt(tf.reduce_mean(stacked_var, axis=0))
    print("Shape ", tf.shape(stacked_var))

    print("Mean: ", mean, " std: ", std)
    return images


def get_aircrafts_test_set():
    data_path_img = "/home/amora/fgvc-aircraft/fgvc-aircraft-2013b/data/images"
    data_path = "/home/amora/fgvc-aircraft/fgvc-aircraft-2013b/data"
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


def get_aircrafts_train_set():
    data_path_img = "/home/amora/fgvc-aircraft/fgvc-aircraft-2013b/data/images"
    data_path = "/home/amora/fgvc-aircraft/fgvc-aircraft-2013b/data"
    train_val_path = os.path.join(data_path, "images_variant_trainval.txt")
    variants_label_path = os.path.join(data_path,
                                       "variants.txt")  # contains all the labels
    convert_label_to_index_dict = {}
    variant_label_id = 0

    # def cut_lower_20px_fn(image):
    #     target_height = tf.shape(image)[0] - 20
    #     target_width = tf.shape(image)[1]
    #     img = tf.image.crop_to_bounding_box(
    #         image, offset_height=0, offset_width=0, target_height=target_height,
    #         target_width=target_width
    #     )
    #     return img

    with open(variants_label_path) as variants_file_txt:
        for line in variants_file_txt:
            variant_as_string = line.split("\n")[0]  # remove the ending line
            # print(variant_as_string)
            convert_label_to_index_dict[variant_as_string] = variant_label_id
            variant_label_id = variant_label_id + 1
    cont = 0
    alphabetical_order = [str(i) for i in range(0, 100)]
    alphabetical_order.sort()
    with open(train_val_path) as train_file_txt:
        for line in train_file_txt:
            split = line.split(maxsplit=1)
            img_file_name = split[0]
            label = convert_label_to_index_dict[split[1].split("\n")[0]]
            folder_name = alphabetical_order[label]
            img_path = os.path.join(data_path_img, img_file_name + ".jpg")
            img = pil_loader(img_path)

            to_save_img = cut_lower_20px_fn(img)
            path = os.path.join("aircrafts_test", "train", folder_name)
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


if __name__ == '__main__':
    # read_images_from_filename(["1236463.jpg"])
    # compute_mean_and_variance()
    # get_aircrafts_test_set()
    get_aircrafts_train_set()
    # label_idx, img_names, labels = fgvc_aircraft_compute_indexes_of_labels()
