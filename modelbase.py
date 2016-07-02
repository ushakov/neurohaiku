import time
import tensorflow as tf
import numpy as np

class Model(object):
    inputs = None
    labels = None
    args = None
    global_step = None
    lr_decay_op = None
    lr = None
    out_labels = None

    PAD_W = 0
    STOP_W = 1

    def __init__(self, args):
        self.args = args

    def prepare_batch(self, seq, training=True):
        pass

    def train(self, sess, saver, train_inputs, valid_inputs):
        losses = []
        reported_losses = []
        summary_writer = None
        if self.args.save:
            summary_writer = tf.train.SummaryWriter(
                self.args.save, flush_secs=10)
        t0 = time.time()
        for i in xrange(self.args.iters):
            loss = self.train_step(sess, train_inputs, summary_writer)
            losses.append(loss)
            step = self.global_step.eval()
            if i == 0 or step % self.args.report_step == 0:
                reported_loss = np.mean(losses)
                losses = []
                if (self.lr_decay_op is not None and 
                    len(reported_losses) > 2 and
                    reported_loss > max(reported_losses[-3:])):
                    sess.run(self.lr_decay_op)
                reported_losses.append(reported_loss)
                
                self.validate_and_report(sess, valid_inputs, summary_writer, 
                                         reported_loss, int(time.time() - t0))

            if (step % 1000 == 0 or i == self.args.iters - 1) and self.args.save:
                saver.save(sess, self.args.save + '/model', global_step=step)
                    
    def train_step(self, sess, inputs, summary_writer):
        pass

    def validate_and_report(self, sess, valid_inputs, summary_writer, train_loss, seconds):
        pass
