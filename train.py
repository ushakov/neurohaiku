# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import random
import argparse
import os
import subprocess
import codecs

class Vocab(object):
    i2w = None
    w2i = None
    
    def __init__(self):
        self.i2w = u'_$^ абвгдежзийклмнопрстуфхцчшщъыьэюя\n'
        self.w2i = {w: i for i, w in enumerate(self.i2w)}

def load_dataset(args, vocab):
    ds = []
    lines = []
    for line in codecs.open(args.filename, 'rt', encoding='utf-8'):
        line = line.strip()
        if not line:
            haiku = u'^{}$'.format('\n'.join(lines))
            ids = []
            for char in haiku:
                assert char in vocab.w2i
                ids.append(vocab.w2i[char])
            if len(ids) > args.length:
                start = 0
                while start + args.length < len(ids) + args.length/2:
                    end = start + args.length
                    if end > len(ids):
                        end = len(ids)
                    ds.append(ids[start:end])
                    start += args.length/2
            else:
                ds.append(ids)
            lines = []
        else:
            lines.append(line)
    print 'loaded', len(ds), 'haiku'
    return ds


def split_into_batches(arr, n):
    start = 0
    while start < len(arr):
        end = start + n
        if end > len(arr):
            end = len(arr)
        yield arr[start:end]
        start = end


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('filename')
    arg('--batch_size', type=int, default=64)
    arg('--hidden_size', type=int, default=50)
    arg('--keep_prob', type=float, default=0.5)
    arg('--l2-coeff', type=float, default=0.0)
    arg('--lr', type=float, default=0.001)
    # arg('--lr-decay', type=float, default=0.9)
    arg('--length', type=int, default=100)
    arg('--vocab_size', type=int, default=None)
    arg('--emb_size', type=int, default=20)
    arg('--iters', type=int, default=30000)
    arg('--report_step', type=int, default=100)
    arg('--generate', action='store_true')
    arg('--save')
    arg('--load')
    # arg('--evaluate', action='store_true')
    arg('--model', default='lstm')
    arg('--max-gradient-norm', type=float, default=5)
    arg('--n-layers', type=int, default=1)
    arg('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    print args
    if args.save:
        timestamp = int(time.time())
        try:
            os.makedirs(args.save)
        except OSError:
            pass
        fname = args.save + '/runparam-' + str(timestamp) + '.txt'
        with open(fname, 'wt') as f:
            f.write('params: ')
            f.write(str(args))
            f.write('\n')
            f.write('git stamp: ')
        subprocess.call('git describe --dirty --always --tags >> ' + fname, shell=True)

    if not args.generate:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    vocab = Vocab()
    args.vocab_size = len(vocab.i2w)

    ds = load_dataset(args, vocab)
    random.shuffle(ds)
    n_val = int(len(ds) * 0.1)
    train, val = ds[n_val:], ds[:n_val]
    print 'train size', len(train)
    print 'val size', len(val)

    args.length = max([len(toks) for toks in ds])
    print 'max length =', args.length

    model_module = __import__(args.model)
    model = model_module.Model(args)

    saver = tf.train.Saver(tf.all_variables())

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        if args.load:
            saver.restore(sess, args.load)
        else:
            sess.run(tf.initialize_all_variables())

        if not args.generate:
            model.train(sess, saver, train, val)
        else:
            seed = u'^'
            ids = model.generate(sess, vocab.w2i[seed])
            res = [seed]
            for id in ids:
                if id == model.STOP_W:
                    break
                res.append(vocab.i2w[id])
            print ''.join(res)
