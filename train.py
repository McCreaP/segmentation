#!/usr/bin/env python3

import logging
import random
import time

import os

import sys
import tensorflow as tf

from batches import InputPreparator, Batches, Phase
from model import Model, MODEL_PATH


PREDICTION_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "predictions")


def main():
    random.seed(42)
    setup_logging()
    # data_path = '../truncated'
    data_path = '/scidata/assignment2/'
    trainer = Trainer(data_path)

    if len(sys.argv) != 2:
        logging.error("Wrong number of arguments. Required: 1, got {0}".format(len(sys.argv) - 1))
        sys.exit(1)

    if sys.argv[1] == 'train':
        phase = Phase.TRAINING
    elif sys.argv[1] == 'validate':
        phase = Phase.VALIDATION
    else:
        logging.error("Invalid argument: {0}".format(sys.argv[1]))
        print_usage()
        sys.exit(1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    config.allow_soft_placement = True
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        if phase == Phase.TRAINING:
            trainer.train(sess)
            save_path = saver.save(sess, MODEL_PATH)
            logging.info("Model saved in path: %s" % save_path)
        elif phase == Phase.VALIDATION:
            saver.restore(sess, MODEL_PATH)
            trainer.report_test_objective(sess)


def print_usage():
    logging.error("Usage: {0} train|validate".format(sys.argv[0]))


def setup_logging():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


class Trainer(object):
    def __init__(self, data_path):
        self.batches = Batches(data_path)
        self.model = Model()

        # Define training on mini batch
        y_conv_flatten = tf.reshape(self.model.y_conv, [-1, 66])
        self.target_one_hot_rescaled = tf.placeholder(tf.float32, [None, 256, 256, 66])
        y_target_flatten = tf.reshape(self.target_one_hot_rescaled, [-1, 66])
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y_conv_flatten, labels=y_target_flatten)
        )
        self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

        # Define validation on a sample
        with tf.device("/cpu:0"):
            self.y_conv_predictions = tf.placeholder(tf.float32, [256, 256, 1])
            # self.y_conv_predictions_flipped = tf.placeholder(tf.float32, [256, 256, 1])
            self.target_original = tf.placeholder(tf.float32, [None, None, 1])

            # y_conv_2 = tf.reverse(self.y_conv_predictions_flipped, axis=[1])
            # y_conv_averaged = (self.y_conv_predictions + y_conv_2) / 2.0
            predictions = InputPreparator.postprocess(self.y_conv_predictions, tf.shape(self.target_original))

            self.accuracy_per_sample = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        predictions,
                        self.target_original
                    ),
                    tf.float32
                )
            )

            self.filename = tf.placeholder(tf.string)
            predictions_png = tf.image.encode_png(
                tf.cast(predictions, tf.uint8)
            )
            self.write_prediction = tf.write_file(self.filename, predictions_png)

    def train(self, sess):
        tf.global_variables_initializer().run()
        num_epochs = 4
        batch_size = 4

        batch_idx = 1
        start_time = time.time()
        try:
            for epoch_idx in range(num_epochs):
                logging.info("Epoch {0}".format(epoch_idx + 1))
                training_data = self.batches.training()
                while training_data.has_next():
                    if batch_idx % 100 == 0:
                        logging.info('Batch {0}'.format(batch_idx))
                    else:
                        logging.debug('Batch {0}'.format(batch_idx))

                    xs, ys = training_data.next_batch(batch_size, sess)
                    end_time = time.time()
                    logging.debug("Batch prepared, time: {0}".format(end_time - start_time))
                    start_time = end_time

                    loss, _ = sess.run([self.loss, self.train_step], feed_dict={self.model.x: xs, self.target_one_hot_rescaled: ys})
                    end_time = time.time()
                    logging.debug("Batch trained, time: {0}".format(end_time - start_time))
                    start_time = end_time
                    if batch_idx % 100 == 0:
                        logging.info('Training loss {0}'.format(loss))
                    if batch_idx % 1000 == 0:
                        self.report_test_objective(sess)

                    batch_idx += 1

        except KeyboardInterrupt:
            logging.info('Stopping training!')
            pass

        self.report_test_objective(sess)

    def report_test_objective(self, sess):
        logging.debug("Start reporting objective")
        validation_data = self.batches.validation()
        batch_size = 4
        accuracy_per_sample = []
        batch_idx = 1
        start_time = time.time()
        while batch_idx <= 100 and validation_data.has_next():
            logging.debug("Processing validation batch: {0}".format(batch_idx))
            batch_idx += 1

            img_ids, xs, labels = validation_data.next_batch(batch_size, sess)
            end_time = time.time()
            logging.debug("Batch prepared, time: {0}".format(end_time - start_time))
            start_time = end_time

            accuracies = self.compute_accuracy_per_sample(img_ids, xs, labels, sess)
            accuracy_per_sample += accuracies
            end_time = time.time()
            logging.debug("Batch predicted, time: {0}".format(end_time - start_time))
            start_time = end_time

        objective = sum(accuracy_per_sample) / len(accuracy_per_sample)
        logging.info('Test objective: {0}'.format(objective))

    def compute_accuracy_per_sample(self, img_ids, batch_xs, batch_labels, sess):
        y_conv_predictions, = sess.run(
            [self.model.y_conv_predictions],
            feed_dict={self.model.x: batch_xs}
        )
        # y_conv_predictions_flipped, = sess.run(
        #     [self.model.y_conv_predictions],
        #     feed_dict={self.model.x: batch_xs_flipped}
        # )
        accuracy_per_sample = []
        for idx, img_id in enumerate(img_ids):
            accuracy, _ = sess.run(
                [self.accuracy_per_sample, self.write_prediction],
                feed_dict={
                    self.y_conv_predictions: y_conv_predictions[idx],
                    # self.y_conv_predictions_flipped: y_conv_predictions_flipped[idx],
                    self.target_original: batch_labels[idx],
                    self.filename: "{0}/{1}.png".format(PREDICTION_ROOT, img_id)
                }
            )
            accuracy_per_sample.append(accuracy)
        return accuracy_per_sample


if __name__ == '__main__':
    main()
