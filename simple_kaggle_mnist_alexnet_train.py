"""
This is simple Alexnet train implementation modified for Kaggle mnist data.
"""

import time
import tensorflow as tf
import numpy as np
import kaggle_mnist_input as loader

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('training_epoch', 10, "training epoch")
tf.app.flags.DEFINE_integer('batch_size', 128, "batch size")
tf.app.flags.DEFINE_integer('validation_interval', 20, "validation interval")

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, "dropout keep prob")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "learning rate")
tf.app.flags.DEFINE_float('momentum', 0.9, "momentum")
tf.app.flags.DEFINE_float('weight_decay', 0.0005, "weight decay")
tf.app.flags.DEFINE_string('train_path', '/tmp/train.csv', "path to download training data")
tf.app.flags.DEFINE_string('test_path', '/tmp/test.csv', "path to download test data")
tf.app.flags.DEFINE_integer('validation_size', 2000, "validation size in training data")

image_size = 28
image_channel = 1
label_cnt = 10

inputs = tf.placeholder("float", [None, image_size, image_size, image_channel])
labels = tf.placeholder("float", [None, label_cnt])
dropout_keep_prob = tf.placeholder("float", None)

# conv layer 1
conv1_weights = tf.Variable(tf.random_normal([7, 7, image_channel, 96], dtype=tf.float32, stddev=0.01))
conv1_biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32))
conv1 = tf.nn.conv2d(inputs, conv1_weights, [1, 3, 3, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, conv1_biases)
conv1_relu = tf.nn.relu(conv1)
conv1_norm = tf.nn.local_response_normalization(conv1_relu, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0)
conv1_pool = tf.nn.max_pool(conv1_norm, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

# conv layer 2
conv2_weights = tf.Variable(tf.random_normal([5, 5, 96, 256], dtype=tf.float32, stddev=0.01))
conv2_biases = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32))
conv2 = tf.nn.conv2d(conv1_pool, conv2_weights, [1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, conv2_biases)
conv2_relu = tf.nn.relu(conv2)
conv2_norm = tf.nn.local_response_normalization(conv2_relu)
conv2_pool = tf.nn.max_pool(conv2_norm, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

# conv layer 3
conv3_weights = tf.Variable(tf.random_normal([3, 3, 256, 384], dtype=tf.float32, stddev=0.01))
conv3_biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32))
conv3 = tf.nn.conv2d(conv2_pool, conv3_weights, [1, 1, 1, 1], padding='SAME')
conv3 = tf.nn.bias_add(conv3, conv3_biases)
conv3_relu = tf.nn.relu(conv3)

# conv layer 4
conv4_weights = tf.Variable(tf.random_normal([3, 3, 384, 384], dtype=tf.float32, stddev=0.01))
conv4_biases = tf.Variable(tf.constant(1.0, shape=[384], dtype=tf.float32))
conv4 = tf.nn.conv2d(conv3_relu, conv4_weights, [1, 1, 1, 1], padding='SAME')
conv4 = tf.nn.bias_add(conv4, conv4_biases)
conv4_relu = tf.nn.relu(conv4)

# conv layer 5
conv5_weights = tf.Variable(tf.random_normal([3, 3, 384, 256], dtype=tf.float32, stddev=0.01))
conv5_biases = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32))
conv5 = tf.nn.conv2d(conv4_relu, conv5_weights, [1, 1, 1, 1], padding='SAME')
conv5 = tf.nn.bias_add(conv5, conv5_biases)
conv5_relu = tf.nn.relu(conv5)
conv5_pool = tf.nn.max_pool(conv5_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# fc layer 1
fc1_weights = tf.Variable(tf.random_normal([256 * 3 * 3, 4096], dtype=tf.float32, stddev=0.01))
fc1_biases = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32))
conv5_reshape = tf.reshape(conv5_pool, [-1, fc1_weights.get_shape().as_list()[0]])
fc1 = tf.matmul(conv5_reshape, fc1_weights)
fc1 = tf.nn.bias_add(fc1, fc1_biases)
fc1_relu = tf.nn.relu(fc1)
fc1_drop = tf.nn.dropout(fc1_relu, dropout_keep_prob)

# fc layer 2
fc2_weights = tf.Variable(tf.random_normal([4096, 4096], dtype=tf.float32, stddev=0.01))
fc2_biases = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32))
fc2 = tf.matmul(fc1_drop, fc2_weights)
fc2 = tf.nn.bias_add(fc2, fc2_biases)
fc2_relu = tf.nn.relu(fc2)
fc2_drop = tf.nn.dropout(fc2_relu, dropout_keep_prob)

# fc layer 3 - output
fc3_weights = tf.Variable(tf.random_normal([4096, label_cnt], dtype=tf.float32, stddev=0.01))
fc3_biases = tf.Variable(tf.constant(1.0, shape=[label_cnt], dtype=tf.float32))
fc3 = tf.matmul(fc2_drop, fc3_weights)
logits = tf.nn.bias_add(fc3, fc3_biases)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

# accuracy
predict = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(labels, 1)), tf.float32))

# train
train = tf.train.RMSPropOptimizer(FLAGS.learning_rate, FLAGS.weight_decay, FLAGS.momentum).minimize(loss)

# session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# load mnist data
train_images, train_labels, train_range, validation_images, validation_labels, validation_indices = loader.load_mnist_train(
    FLAGS.validation_size, FLAGS.batch_size)

# begin training
train_start_time = time.time()
total_train_len = len(train_images)
i = 0
for epoch in range(FLAGS.training_epoch):
    epoch_start_time = time.time()
    for start, end in train_range:
        batch_start_time = time.time()
        trainX = train_images[start:end]
        trainY = train_labels[start:end]
        _, loss_result = sess.run([train, loss], feed_dict={inputs: trainX, labels: trainY,
                                                            dropout_keep_prob: FLAGS.dropout_keep_prob})
        print('[%s][training][epoch %d, step %d exec %.2f seconds] [file: %5d ~ %5d / %5d] loss : %3.3f' % (
            time.strftime("%Y-%m-%d %H:%M:%S"), epoch, i, (time.time() - batch_start_time), start, end,
            total_train_len, loss_result))

        if i % FLAGS.validation_interval == 0 and i > 0:
            validation_start_time = time.time()
            shuffle_indices = loader.shuffle_validation(validation_indices, FLAGS.batch_size)
            testX = validation_images[shuffle_indices]
            testY = validation_labels[shuffle_indices]
            accuracy_result, loss_result = sess.run([accuracy, loss],
                                                    feed_dict={inputs: testX, labels: testY, dropout_keep_prob: 1.0})
            print('[%s][test][epoch %d, step %d exec %.2f seconds] accuracy : %1.3f, loss : %3.3f' % (
                time.strftime("%Y-%m-%d %H:%M:%S"), epoch, i, (time.time() - validation_start_time),
                accuracy_result, loss_result))
                
        i += 1

    print("[%s][epoch exec %s seconds] epoch : %d" % (
        time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - epoch_start_time), epoch))

print("[%s][total exec %s seconds" % (time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - train_start_time)))
