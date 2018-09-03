'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c)  2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("..")
from modules.sequential import Sequential
from modules.convolution import Convolution
from modules.utils import Utils

import tensorflow as tf
import numpy as np
import scipy.io as sio

mat_data=sio.loadmat('./Data_1')
train_data=mat_data['train_data']
train_labels=mat_data['train_labels']
test_data=mat_data['test_data']
test_labels=mat_data['test_labels']

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 100, 'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 100, 'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 1, 'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.0001, 'Initial learning rate')
flags.DEFINE_string("data_dir", 'data', 'Directory for storing data')
flags.DEFINE_string("summaries_dir", 'ex_logs', 'Summaries directory')
flags.DEFINE_boolean("relevance", True, 'Compute relevances')
flags.DEFINE_string("relevance_method", 'simple', 'relevance methods: simple/eps/w^2/alphabeta')
flags.DEFINE_boolean("save_model", False, 'Save the trained model')
flags.DEFINE_boolean("reload_model", False, 'Restore the trained model')
flags.DEFINE_string("checkpoint_dir", 'mnist_trained_model', 'Checkpoint dir')
flags.DEFINE_string("checkpoint_reload_dir", 'mnist_trained_model', 'Checkpoint dir')
FLAGS = flags.FLAGS

def nn():
    return Sequential([Convolution(output_depth=36,input_depth=1,batch_size=FLAGS.batch_size, input_dim=25, act ='relu', stride_size=1, pad='VALID'),
                       Convolution(output_depth=25, kernel_size=5, stride_size=3, act='relu', pad='VALID'),
                       Convolution(output_depth=16, kernel_size=5, stride_size=2, act='relu', pad='VALID'),
                       Convolution(output_depth=2, kernel_size=1, stride_size=1, act='relu', pad='VALID')
                       ])
trX, trY, teX, teY = train_data, train_labels, test_data, test_labels

def train():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 25, 25], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
            keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('model'):
            net = nn()
            inp = tf.pad(tf.reshape(x, [FLAGS.batch_size, 25, 25, 1]), [[0, 0], [0, 0], [0, 0], [0, 0]])
            op = net.forward(inp)
            y = tf.squeeze(op)
            trainer = net.fit(output=y, ground_truth=y_, loss='softmax_crossentropy', optimizer='adam', opt_params=[FLAGS.learning_rate])
        with tf.variable_scope('relevance'):
            if FLAGS.relevance:
                LRP = net.lrp(op, FLAGS.relevance_method, 1e-8)
            else:
                LRP = []

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
        tf.global_variables_initializer().run()
        utils = Utils(sess, FLAGS.checkpoint_dir)

        if FLAGS.reload_model:
            utils.reload_model()

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost)
        predict_op = tf.argmax(y, 1)
        tf.initialize_all_variables().run()

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            for i in range(1000):
                sess.run(train_op, feed_dict={x: trX, y_: trY})
                if 100 * sess.run(accuracy, feed_dict={x: trX, y_: trY})==100:
                    print(100 * np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={x: teX, y_: teY})))
            for bnum in range(int(len(trX) / FLAGS.batch_size)):
                test_inp = {x: teX, y_: teY}
                relevance_test = sess.run(LRP, feed_dict={x: teX, y_: teY})
                relevance_category = sess.run(predict_op, feed_dict={x: teX, y_: teY})
                relevance_truth = np.argmax(teY, axis=1)
                accuracy = 100 * np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={x: teX, y_: teY}))

        if FLAGS.relevance:
            sio.savemat('LRP_result.mat',
                        {"relevance_test": relevance_test,
                         "relevance_category": relevance_category,
                         "relevance_truth": relevance_truth,
                         "accuracy": accuracy
                         })

        train_writer.close()
        test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()