import tensorflow as tf
import os
import kaggle_mnist_alexnet_model as model
import kaggle_mnist_input as loader
import time
import csv

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('training_epoch', 40, "training epoch")
tf.app.flags.DEFINE_integer('batch_size', 128, "batch size")
tf.app.flags.DEFINE_integer('validation_interval', 20, "validation interval")

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, "dropout keep prob")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "learning rate")
tf.app.flags.DEFINE_float('rms_decay', 0.9, "rms optimizer decay")
tf.app.flags.DEFINE_float('weight_decay', 0.0005, "l2 regularization weight decay")
tf.app.flags.DEFINE_string('train_path', '/tmp/train.csv', "path to download training data")
tf.app.flags.DEFINE_string('test_path', '/tmp/test.csv', "path to download test data")
tf.app.flags.DEFINE_integer('validation_size', 2000, "validation size in training data")
tf.app.flags.DEFINE_string('save_name', os.getcwd() + '/var.ckpt', "path to save variables")
tf.app.flags.DEFINE_boolean('is_train', True, "True for train, False for test")
tf.app.flags.DEFINE_string('test_result', 'result.csv', "test file path")

image_size = 28
image_channel = 1
label_cnt = 10


def train():
    # build graph
    inputs, labels, dropout_keep_prob, learning_rate = model.input_placeholder(image_size, image_channel, label_cnt)
    logits = model.inference(inputs, dropout_keep_prob, label_cnt)

    accuracy = model.accuracy(logits, labels)
    loss = model.loss(logits, labels)
    train = tf.train.RMSPropOptimizer(learning_rate, FLAGS.rms_decay).minimize(loss)

    # session
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # ready for summary
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('./summary/train', sess.graph)
    validation_writer = tf.train.SummaryWriter('./summary/validation')

    # tf saver
    saver = tf.train.Saver()
    if os.path.isfile(FLAGS.save_name):
        saver.restore(sess, FLAGS.save_name)

    total_start_time = time.time()

    # load mnist data
    train_images, train_labels, train_range, validation_images, validation_labels, validation_indices = loader.load_mnist_train(
        FLAGS.validation_size, FLAGS.batch_size)

    total_train_len = len(train_images)
    i = 0
    cur_learning_rate = FLAGS.learning_rate
    for epoch in range(FLAGS.training_epoch):
        if epoch % 10 == 0 and epoch > 0:
            cur_learning_rate /= 10
        epoch_start_time = time.time()
        for start, end in train_range:
            batch_start_time = time.time()
            train_x = train_images[start:end]
            train_y = train_labels[start:end]
            if i % 20 == 0:
                summary, _, loss_result = sess.run([merged, train, loss], feed_dict={inputs: train_x, labels: train_y,
                                                                                     dropout_keep_prob: FLAGS.dropout_keep_prob,
                                                                                     learning_rate: cur_learning_rate})
                train_writer.add_summary(summary, i)
            else:
                _, loss_result = sess.run([train, loss], feed_dict={inputs: train_x, labels: train_y,
                                                                    dropout_keep_prob: FLAGS.dropout_keep_prob,
                                                                    learning_rate: cur_learning_rate})
            print('[%s][training][epoch %d, step %d exec %.2f seconds] [file: %5d ~ %5d / %5d] loss : %3.10f' % (
                time.strftime("%Y-%m-%d %H:%M:%S"), epoch, i, (time.time() - batch_start_time), start, end,
                total_train_len, loss_result))

            if i % FLAGS.validation_interval == 0 and i > 0:
                validation_start_time = time.time()
                shuffle_indices = loader.shuffle_validation(validation_indices, FLAGS.batch_size)
                validation_x = validation_images[shuffle_indices]
                validation_y = validation_labels[shuffle_indices]
                summary, accuracy_result, loss_result = sess.run([merged, accuracy, loss],
                                                                 feed_dict={inputs: validation_x, labels: validation_y,
                                                                            dropout_keep_prob: 1.0})
                validation_writer.add_summary(summary, i)
                print('[%s][validation][epoch %d, step %d exec %.2f seconds] accuracy : %1.3f, loss : %3.10f' % (
                    time.strftime("%Y-%m-%d %H:%M:%S"), epoch, i, (time.time() - validation_start_time),
                    accuracy_result, loss_result))

            i += 1

        print("[%s][epoch exec %s seconds] epoch : %d" % (
            time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - epoch_start_time), epoch))
        saver.save(sess, FLAGS.save_name)
    print("[%s][total exec %s seconds" % (time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - total_start_time)))
    train_writer.close()
    validation_writer.close()


def test():
    # build graph
    inputs, labels, dropout_keep_prob, learning_rate = model.input_placeholder(FLAGS.image_size, FLAGS.image_channel,
                                                                               FLAGS.label_cnt)
    logits = model.inference(inputs, dropout_keep_prob)
    predict = tf.argmax(logits, 1)

    # session
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # tf saver
    saver = tf.train.Saver()
    if os.path.isfile(FLAGS.save_name):
        saver.restore(sess, FLAGS.save_name)

    i = 1

    # load test data
    test_images, test_ranges = loader.load_mnist_test(FLAGS.batch_size)

    # ready for result file
    test_result_file = open(FLAGS.test_result, 'wb')
    csv_writer = csv.writer(test_result_file)
    csv_writer.writerow(['ImageId', 'Label'])

    total_start_time = time.time()

    for file_start, file_end in test_ranges:
        test_x = test_images[file_start:file_end]
        predict_label = sess.run(predict, feed_dict={inputs: test_x, dropout_keep_prob: 1.0})

        for cur_predict in predict_label:
            csv_writer.writerow([i, cur_predict])
            print('[Result %s: %s]' % (i, cur_predict))
            i += 1
    print("[%s][total exec %s seconds" % (time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - total_start_time)))


def main(_):
    if FLAGS.is_train:
        train()
    else:
        test()


if __name__ == '__main__':
    tf.app.run()
