import torch
from torch import optim

import os
import os.path
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

import utilsFast
from utilsFast import read_vocab, Tokenizer, timeSince, try_cuda
from env import R2RBatch, ImageFeatures
from modelFast import SpeakerEncoderLSTM, SpeakerDecoderLSTM
from speaker import Seq2SeqSpeaker

from vocab import SUBTRAIN_VOCAB, TRAIN_VOCAB, TRAINVAL_VOCAB

MAX_INSTRUCTION_LENGTH = 80

batch_size = 100
max_episode_len = 10
word_embedding_size = 300
glove_path = 'tasks/R2R/data/train_glove.npy'
action_embedding_size = 2048+128
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
feedback_method = 'teacher'  # teacher or sample
learning_rate = 0.0001
weight_decay = 0.0005
FEATURE_SIZE = 2048+128
n_iters = 20000
log_every = 100
save_every = 1000

def make_speaker(args):
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    glove = np.load(glove_path)
    feature_size = FEATURE_SIZE
    vocab = read_vocab(TRAIN_VOCAB)
    encoder = try_cuda(SpeakerEncoderLSTM(
        action_embedding_size, feature_size, enc_hidden_size, dropout_ratio,
        bidirectional=bidirectional))
    decoder = try_cuda(SpeakerDecoderLSTM(
        len(vocab), word_embedding_size, hidden_size, dropout_ratio,
        glove=glove))
    agent = Seq2SeqSpeaker(
        None, "", encoder, decoder, MAX_INSTRUCTION_LENGTH)
    return agent

