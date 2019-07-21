import os
import random
from enum import Enum

import tensorflow as tf
import logging


class LocationHolder:
    def __init__(self, root):
        self.training_path = '{0}/training'.format(root)
        self.x_path = '{0}/images'.format(self.training_path)
        self.y_path = '{0}/labels_plain'.format(self.training_path)

    def get_img_path(self, img_id):
        return "{0}/{1}.jpg".format(self.x_path, img_id)

    def get_labels_path(self, img_id):
        return "{0}/{1}.png".format(self.y_path, img_id)


class Batches:
    def __init__(self, root):
        self.location_holder = LocationHolder(root)
        self.input_data = InputData(self.location_holder)

    def training(self):
        return BatchIter(self.location_holder, self.input_data.training(), Phase.TRAINING)

    def validation(self):
        return BatchIter(self.location_holder, self.input_data.validation(), Phase.VALIDATION)


class BatchIter:
    def __init__(self, location_holder, input_data_iter, phase):
        self.input_data_iter = input_data_iter
        self.input_preparator = InputPreparator(location_holder, phase)

    def next_batch(self, batch_size, sess):
        sample_ids = self.input_data_iter.next_batch(batch_size)
        return self.input_preparator.prepare(sample_ids, sess)

    def has_next(self):
        return self.input_data_iter.has_next()


class InputData:
    def __init__(self, location_holder):
        ids = [img[:-4] for img in os.listdir(location_holder.x_path)]
        self.train_ids, self.validation_ids = self.split(ids, train_ratio=0.9)
        logging.info("Training set size: {0}, validation set size: {1}".format(len(self.train_ids), len(self.validation_ids)))

    @staticmethod
    def split(ids, train_ratio):
        train_size = int(train_ratio * len(ids))
        random.shuffle(ids)
        return ids[:train_size], ids[train_size:]

    def training(self):
        random.shuffle(self.train_ids)
        return InputDataIter(self.train_ids)

    def validation(self):
        return InputDataIter(self.validation_ids)


class InputDataIter:
    def __init__(self, sample_ids):
        self.sample_ids = sample_ids
        self.curr_idx = 0

    def next_batch(self, batch_size):
        end = min(self.curr_idx + batch_size, len(self.sample_ids))
        batch_ids = self.sample_ids[self.curr_idx:end]
        self.curr_idx += len(batch_ids)
        return batch_ids

    def has_next(self):
        return self.curr_idx < len(self.sample_ids)


class Phase(Enum):
    TRAINING = 1
    VALIDATION = 2


class InputPreparator:
    def __init__(self, location_holder, phase):
        self.location_holder = location_holder
        self.phase = phase

        with tf.device("/cpu:0"):
            self.horizontal_flip = tf.placeholder(tf.bool)

            # Define image preprocessing
            self.img_path = tf.placeholder(tf.string)
            image_str = tf.read_file(self.img_path)
            image = tf.image.decode_jpeg(image_str)
            augumented_image = tf.cond(
                self.horizontal_flip,
                true_fn=lambda: tf.image.flip_left_right(image),
                false_fn=lambda: image
            )
            self.x = tf.image.resize_images(augumented_image, [256, 256])

            # Define labels decoding
            self.labels_path = tf.placeholder(tf.string)
            labels_str = tf.read_file(self.labels_path)
            original_labels = tf.image.decode_png(labels_str)
            self.labels = tf.cond(
                self.horizontal_flip,
                true_fn=lambda: tf.image.flip_left_right(original_labels),
                false_fn=lambda: original_labels
            )

            # Define labels preprocessing
            resized = tf.cast(
                tf.image.resize_images(self.labels, [256, 256]),
                tf.int32
            )
            reshaped = tf.reshape(resized, [256, 256])  # Reduce rank, there is only one channel
            self.y = tf.one_hot(reshaped, depth=66)

    def prepare(self, sample_ids, sess):
        if self.phase == Phase.TRAINING:
            return self.prepare_training(sample_ids, sess)
        else:
            return self.prepare_validation(sample_ids, sess)

    def prepare_training(self, sample_ids, sess):
        xs = []
        ys = []
        for sample_id in sample_ids:
            x, y = sess.run(
                [self.x, self.y],
                feed_dict={
                    self.img_path: self.location_holder.get_img_path(sample_id),
                    self.labels_path: self.location_holder.get_labels_path(sample_id),
                    self.horizontal_flip: random.random() < 0.5
                }
            )
            xs.append(x)
            ys.append(y)
        return xs, ys

    def prepare_validation(self, sample_ids, sess):
        xs = []
        # xs_flipped = []
        batch_labels = []
        for sample_id in sample_ids:
            x, labels = sess.run(
                [self.x, self.labels],
                feed_dict={
                    self.img_path: self.location_holder.get_img_path(sample_id),
                    self.labels_path: self.location_holder.get_labels_path(sample_id),
                    self.horizontal_flip: False
                }
            )
            # x_flipped, = sess.run(
            #     [self.x],
            #     feed_dict={
            #         self.img_path: self.location_holder.get_img_path(sample_id),
            #         self.horizontal_flip: True
            #     }
            # )
            xs.append(x)
            # xs_flipped.append(x_flipped)
            batch_labels.append(labels)
        return sample_ids, xs, batch_labels

    @staticmethod
    def postprocess(predictions_nn, original_shape):
        return tf.image.resize_images(predictions_nn, [original_shape[0], original_shape[1]])
