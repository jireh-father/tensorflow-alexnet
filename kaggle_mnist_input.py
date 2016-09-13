import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
from six.moves import urllib

FLAGS = tf.app.flags.FLAGS
TRAIN_DATA_URL = 'http://file.hovits.com/dl/train.csv'
TEST_DATA_URL = 'http://file.hovits.com/dl/teset.csv'


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mnist_train(validation_size=2000, batch_size=128):
    download_train()

    data = pd.read_csv(FLAGS.train_path)

    images = data.iloc[:, 1:].values
    images = images.astype(np.float)

    images = np.multiply(images, 1.0 / 255.0)

    image_size = images.shape[1]

    image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
    images = images.reshape(-1, image_width, image_height, 1)

    labels_flat = data[[0]].values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]

    train_range = zip(range(0, len(train_images), batch_size), range(batch_size, len(train_images), batch_size))

    if len(train_images) % batch_size > 0:
        train_range.append((train_range[-1][1], len(train_images)))

    validation_indices = np.arange(len(validation_images))

    return train_images, train_labels, train_range, validation_images, validation_labels, validation_indices


def shuffle_validation(validation_indices, batch_size):
    np.random.shuffle(validation_indices)
    return validation_indices[0:batch_size]


def download_train():
    statinfo = download(FLAGS.train_path, TRAIN_DATA_URL)
    if statinfo:
        print('Training data is successfully downloaded', statinfo.st_size, 'bytes.')
    else:
        print('Training data was already downloaded')


def download_test():
    statinfo = download(FLAGS.test_path, TEST_DATA_URL)
    if statinfo:
        print('Test data is successfully downloaded', statinfo.st_size, 'bytes.')
    else:
        print('Test data was already downloaded')


def download(path, url):
    if not os.path.exists(path):
        if not os.path.isdir(os.path.basename(path)):
            os.makedirs(os.path.basename(path))

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (path, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        file_path, _ = urllib.request.urlretrieve(url, path, _progress)
        print()
        return os.stat(file_path)
