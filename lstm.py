#!/usr/bin/env python
# encoding: utf-8

import random
import time
import numpy as np
import tensorflow as tf

import modelbase


class Model(modelbase.Model):
    def __init__(self, args):
        self.args = args

        self.inputs = tf.placeholder(tf.int32, [None, args.length])
        self.true_outputs = tf.placeholder(tf.int32, [None, args.length])
        self.keep_prob = tf.placeholder(tf.float32)

        cell = tf.nn.rnn_cell.LSTMCell(args.hidden_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        if args.n_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.n_layers, state_is_tuple=True)

        self.embedding = tf.get_variable("embedding", [args.vocab_size, args.emb_size])
        embedded = tf.nn.embedding_lookup(self.embedding, self.inputs)
        embedded = [tf.squeeze(input_, [1])
                    for input_ in tf.split(1, args.length, embedded)]

        with tf.variable_scope('labels'):
            softmax_W = tf.get_variable('W', [args.hidden_size, args.vocab_size],
                                        initializer=tf.random_normal_initializer())
            softmax_b = tf.get_variable('b', [args.vocab_size],
                                        initializer=tf.constant_initializer(0.0))

        outputs, _ = tf.nn.rnn(cell, inputs = embedded, dtype=tf.float32)

        self.output_probs = [(tf.matmul(out, softmax_W) + softmax_b) for out in outputs]
        self.outputs = [tf.argmax(prob, 1) for prob in self.output_probs]

        softmax = []
        for i in range(self.args.length-1):
            softmax.append(tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.output_probs[i], self.true_outputs[:,i],
                name='seq_loss_{}'.format(i)))
        self.loss = (1./self.args.length) * tf.reduce_mean(tf.add_n(softmax))
        self.l2norm = sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss + args.l2_coeff * self.l2norm, params)
        gradients, _norm = tf.clip_by_global_norm(gradients, self.args.max_gradient_norm)

        self.train_op = optimizer.apply_gradients(
            zip(gradients, params), global_step=self.global_step)

    def prepare_batch(self, inputs, training=True):
        batch_size = len(inputs)

        input = np.zeros([batch_size, self.args.length], dtype=np.int32)
        output = np.zeros([batch_size, self.args.length], dtype=np.int64)
        for n_batch, input_ids in enumerate(inputs):
            if len(input_ids) > self.args.length:
                input_ids = input_ids[:self.args.length]
            n_pad = self.args.length - len(input_ids)
            padded_input = input_ids + [self.PAD_W] * n_pad
            padded_output = padded_input[1:] + [self.STOP_W]
            for i, w in enumerate(padded_input):
                input[n_batch, i] = w
            for i, w in enumerate(padded_output):
                output[n_batch, i] = w

        feed_dict = {
            self.inputs: input,
            self.true_outputs: output,
        }
        if training:
            feed_dict[self.keep_prob] = self.args.keep_prob
        else:
            feed_dict[self.keep_prob] = 1
        return feed_dict

    def train_step(self, sess, inputs, summary_writer):
        b_inputs = [random.choice(inputs) for _ in xrange(self.args.batch_size)]
        feed_dict = self.prepare_batch(b_inputs, True)
        ops = [self.loss, self.train_op]
        loss, _ = sess.run(ops, feed_dict)
        step = self.global_step.eval()
        return loss

    def validate_and_report(self, sess, valid_inputs, summary_writer, train_loss, seconds):
        feed_dict = self.prepare_batch(valid_inputs, False)
        val_loss = sess.run([self.loss], feed_dict=feed_dict)[0]
        print '{:>5}: train loss {:.4f}, valid loss {:.4f} in {}s'\
            .format(
                self.global_step.eval(),
                train_loss,
                val_loss,
                seconds)
