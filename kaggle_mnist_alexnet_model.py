import tensorflow as tf
import ops as op


def input_placeholder(image_size, image_channel, label_cnt):
    with tf.name_scope('inputlayer'):
        inputs = tf.placeholder("float", [None, image_size, image_size, image_channel], 'inputs')
        labels = tf.placeholder("float", [None, label_cnt], 'labels')
    dropout_keep_prob = tf.placeholder("float", None, 'keep_prob')
    learning_rate = tf.placeholder("float", None, name='learning_rate')

    return inputs, labels, dropout_keep_prob, learning_rate


def inference(inputs, dropout_keep_prob, label_cnt):
    # todo: change lrn parameters
    # conv layer 1
    with tf.name_scope('conv1layer'):
        conv1 = op.conv(inputs, 7, 96, 3)
        conv1 = op.lrn(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

    # conv layer 2
    with tf.name_scope('conv2layer'):
        conv2 = op.conv(conv1, 5, 256, 1, 1.0)
        conv2 = op.lrn(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

    # conv layer 3
    with tf.name_scope('conv3layer'):
        conv3 = op.conv(conv2, 3, 384, 1)

    # conv layer 4
    with tf.name_scope('conv4layer'):
        conv4 = op.conv(conv3, 3, 384, 1, 1.0)

    # conv layer 5
    with tf.name_scope('conv5layer'):
        conv5 = op.conv(conv4, 3, 256, 1, 1.0)
        conv5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # fc layer 1
    with tf.name_scope('fc1layer'):
        fc1 = op.fc(conv5, 4096, 1.0)
        fc1 = tf.nn.dropout(fc1, dropout_keep_prob)

    # fc layer 2
    with tf.name_scope('fc2layer'):
        fc2 = op.fc(fc1, 4096, 1.0)
        fc2 = tf.nn.dropout(fc2, dropout_keep_prob)

    # fc layer 3 - output
    with tf.name_scope('fc3layer'):
        return op.fc(fc2, label_cnt, 1.0, None)


def accuracy(logits, labels):
    # accuracy
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
        tf.scalar_summary('accuracy', accuracy)
    return accuracy


def loss(logits, labels):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        tf.scalar_summary('loss', loss)
    return loss


def train_rms_prop(loss, learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp'):
    return tf.train.RMSPropOptimizer(learning_rate, decay, momentum, epsilon, use_locking, name).minimize(loss)
