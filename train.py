#!/usr/bin/env python
# encoding: utf-8
import argparse
import os
from model import Model


def _main(args):
    train_input_file = os.path.join(args.corpus_dir, '/train/in.txt')
    train_output_file = os.path.join(args.corpus_dir, '/train/out.txt')
    test_input_file = os.path.join(args.corpus_dir, '/test/in.txt')
    test_output_file = os.path.join(args.corpus_dir, '/test/out.txt')
    vocab_file = os.path.join(args.corpus_dir, 'vocabs')
    output_dir = args.output_dir
    if not args.output_dir:
        output_dir = os.path.join(args.corpus_dir, 'output_dir')

    m = Model(
        train_input_file,
        train_output_file,
        test_input_file,
        test_output_file,
        vocab_file,
        num_units=args.num_units, layers=args.layers, dropout=args.dropout,
        save_step=args.save_step, eval_step=args.eval_step,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        output_dir=output_dir,
        restore_model=args.restore_model,
        decode_method=args.decode_method
    )
    m.train(args.train_steps)


def add_arguments(parser):
    parser.add_argument("corpus_dir", type=str, default=None,
                        help="corpus dir should include train/test dir and vocabs file")
    parser.add_argument("--num_units", type=int, default=768,
                        help="")
    parser.add_argument("--layers", type=int, default=4, help="")
    parser.add_argument("--dropout", type=float, default=0.2, help="")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="")
    parser.add_argument("--save_step", type=int, default=10000,
                        help="save checkpoint model every save_step")
    parser.add_argument("--eval_step", type=int, default=1000, help="eval model every eval_step")
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Store log/model files.")
    parser.add_argument("--restore_model", type=bool, default=None,
                        help="restore model from checkpoint")
    parser.add_argument("--train_steps", type=int, default=200000,
                        help="steps to train")
    parser.add_argument("--decode_method", type=str, default='greedy',
                        help="greedy|beam")


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    add_arguments(_parser)
    _args = _parser.parse_args()
    _main(_args)
