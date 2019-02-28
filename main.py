#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import yaml
import pickle as pkl

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext
from torchtext.data import Field

import pdb
import gensim
import numpy as np

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
### Vec mods ###
# from seq2seq.models import VecDecoderRNN
### Done ###
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

if torch.cuda.is_available():
    torch.device('cuda')
else:
    torch.device('cpu')

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

def add_vectors(exes, input_vocab=None, word_vectors=None, vector_features=False):
    if word_vectors is not None:
        data_vectors = t.build_vec_batch(
            input_vocab, [ex.src for ex in exes], word_vectors)
    else:
        data_vectors = np.zeros((len(exes), 0))

    if vector_features:
        vecSrcs = [ex.vector for ex in exes]
        vecTargs = [ex.vecTarg for ex in exes]
        # s|t x batch x vdim
        fvecs = np.array([vecSrcs, vecTargs])
        # rearrange to batch x [vdim x s|t]
        fvecs = np.swapaxes(fvecs, 0, 1)
        fvecs = np.reshape(fvecs, (fvecs.shape[0], 
                                   fvecs.shape[1] * fvecs.shape[2]))
        # concat to other representation
        data_vectors = np.hstack([data_vectors, fvecs])

    for ex, vi in zip(exes.examples, data_vectors):
        ex.vector = vi

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

# TODO I don't think this is best way to load configs
# Remove default values
# load arguments from config file
# set overrides if they're specified in the command line
with open(sys.argv[2]) as f:
    # with open('config.yml') as f:
    config = yaml.safe_load(f)

# TODO make sure these entirely match up with config.yml file
parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', dest='config',
                    default=None,
                    help='Path to config file')
parser.add_argument('--train_path', action='store', dest='train_path',
                    default=config['train_path'],
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    default=config['valid_path'],
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    default=config['load_model'],
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
parser.add_argument("--vector-features", dest="vector_features",
                    action="store_true",
                    default=config["vector_features"],
                    help="If true, grammatical features are provided as dense vectors vecSrc and vecTarg. Input must be in JSON format.")

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

## ADDING VECTORS ###
if config['use_vecs']:
    # train_vecs = load_vec(opt.train_path, config['batch size'])
    # dev_vecs = load_vec(opt.dev_path, config['batch size'])
    print("Loading word vectors")
    vectors = gensim.models.KeyedVectors.load_word2vec_format(config['vecs'], binary=True)
    print("Vectors loaded")
else:
    # train_vecs = None
    # dev_vecs = None
    vectors = None
# pdb.set_trace()

use_vectors = opt.vector_features or (vectors is not None)

# For building the datasets:
max_len = config['max_length']
def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

def load_vec(path, batch_size):
    vec_path = path.replace('data.txt', 'vectors.pkl')
    vecs = pkl.load(open(vec_path, 'rb'))
    # for example, v in zip(torchdata, vecs):
    #     example.vec = v
    batched = []
    for b in range((len(vecs) // batch_size) + 1):
        # pdb.set_trace()
        bat = []
        while len(bat) < batch_size:
            if len(vecs) > 0:
                # try:
                bat.append(vecs.pop(0))
                # except:
                #     pdb.set_trace()
            else:
                break
        batched.append(bat)
    return batched

# If selected, load word vectors:
# if config['vectors']:
#     vectors = gensim.models.KeyedVectors.load_word2vec_format(config['vectors'], binary=True)
#     vec_dim = vectors['vec_size']
# else:
#     vectors = None

# HACKY CHECK
if opt.load_checkpoint == 'None':
    opt.load_checkpoint = None

if opt.load_checkpoint is not None:
    # opt.load_checkpoint = str(opt.load_checkpoint)
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    if not opt.vector_features:
        # Prepare dataset
        src = SourceField()
        tgt = TargetField()
        train = torchtext.data.TabularDataset(
            path=opt.train_path, format='tsv',
            fields=[('src', src), ('tgt', tgt)],
            filter_pred=len_filter
        )
        dev = torchtext.data.TabularDataset(
            path=opt.dev_path, format='tsv',
            fields=[('src', src), ('tgt', tgt)],
            filter_pred=len_filter
        )
    else:
        # load JSON dataset with control vectors
        print("Vector feature mode on.")
        src = SourceField()
        tgt = TargetField()
        vecSrc = Field(sequential=False, use_vocab=False,
                       batch_first=True, dtype=torch.float)
        vecTarg = Field(sequential=False, use_vocab=False,
                       batch_first=True, dtype=torch.float)

        train = torchtext.data.TabularDataset(
            path=opt.train_path, format='json',
            fields={ "src" : ('src', src),
                     "targ" : ('tgt', tgt),
                     "vecSrc" :  ("vector", vecSrc),
                     "vecTarg" :  ("vecTarg", vecTarg),
                 },
            filter_pred=len_filter
        )
        dev = torchtext.data.TabularDataset(
            path=opt.dev_path, format='json',
            fields={ "src" : ('src', src),
                     "targ" : ('tgt', tgt),
                     "vecSrc" :  ("vector", vecSrc),
                     "vecTarg" :  ("vecTarg", vecTarg),
                 },
            filter_pred=len_filter
        )

    src.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # pdb.set_trace()

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = config['encoder embed']
        bidirectional = True
        encoder = EncoderRNN(len(src.vocab), 
                             max_len, 
                             hidden_size, 
                             bidirectional=bidirectional, 
                             rnn_cell='LSTM', 
                             variable_lengths=True)
        # pdb.set_trace()
        # if config['use_vecs']:
        #     decoder = VecDecoderRNN(len(tgt.vocab),
        #                             max_len,
        #                             hidden_size * 2 if bidirectional else hidden_size,
        #                             dropout_p=float(config['dropout']),
        #                             use_attention=True,
        #                             bidirectional=bidirectional,
        #                             rnn_cell='LSTM',
        #                             eos_id=tgt.eos_id,
        #                             sos_id=tgt.sos_id)
        # else:
        if bidirectional:
            hidden_size = hidden_size * 2
        if config['use_vecs']:
            # aug_size = len(train_vecs[0][0])
            aug_size = vectors.vector_size
        else:
            aug_size = 0

        if opt.vector_features:
            featureDim = len(train.examples[0].vector)
            aug_size += 2 * featureDim

        # pdb.set_trace()
        decoder = DecoderRNN(len(tgt.vocab),
                             max_len,
                             hidden_size=hidden_size,
                             aug_size=aug_size,
                             dropout_p=float(config['dropout']),
                             use_attention=True,
                             bidirectional=bidirectional,
                             rnn_cell='LSTM',
                             eos_id=tgt.eos_id,
                             sos_id=tgt.sos_id)
        # if torch.cuda.is_available():
        #     encoder.cuda()
        #     decoder.cuda()
        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss,
                          batch_size=config['batch size'],
                          checkpoint_every=config['checkpoint_every'],
                          print_every=config['print every'],
                          expt_dir=config['expt_dir'])
                          # expt_dir=opt.expt_dir)

    # TODO add dev eval here for early stopping
    if config['train model']:
        add_vectors(train, input_vocab, vectors, opt.vector_features)
        add_vectors(dev, input_vocab, vectors, opt.vector_features)

        # pdb.set_trace()
        seq2seq = t.train(input_vocab,
                          seq2seq, train,
                          num_epochs=config['epochs'],
                          vectors=use_vectors,
                          dev_data=dev,
                          optimizer=optimizer,
                          teacher_forcing_ratio=0.5,
                          resume=opt.resume)

predictor = Predictor(seq2seq, input_vocab, output_vocab)

if config['eval val']:
    of = open(config['output'], 'w')
    # TODO add option to save output
    correct = 0
    total = 0
    if not opt.vector_features:
        src = SourceField()
        tgt = TargetField()
        dev = torchtext.data.TabularDataset(
            path=opt.dev_path, format='tsv',
            fields=[('src', src), ('tgt', tgt)],
            filter_pred=len_filter
        )
    else:
        # load JSON dataset with control vectors
        print("Vector feature mode on.")
        src = SourceField()
        tgt = TargetField()
        vecSrc = Field(sequential=False, use_vocab=False,
                       batch_first=True, dtype=torch.float)
        vecTarg = Field(sequential=False, use_vocab=False,
                       batch_first=True, dtype=torch.float)

        dev = torchtext.data.TabularDataset(
            path=opt.dev_path, format='json',
            fields={ "src" : ('src', src),
                     "targ" : ('tgt', tgt),
                     "vecSrc" :  ("vector", vecSrc),
                     "vecTarg" :  ("vecTarg", vecTarg),
                 },
            filter_pred=len_filter
        )

    add_vectors(dev, input_vocab, vectors, opt.vector_features)

    for ex in dev.examples:
        guess = predictor.predict(ex.src, ex.vector)
        # [1:] tgt starts with <bos> but model output doesn't
        if guess == ex.tgt[1:]:
            correct += 1
        total += 1
        of.write('\t'.join([' '.join(ex.src), ' '.join(ex.tgt[1:-1]), ' '.join(guess[0:-1])]) + '\n')
    print("Val accuracy:", correct, "out of", total, correct / total)


# old interactive testing env
if config['interact']:
    while True:
        seq_str = raw_input("Type in a source sequence:")
        seq = seq_str.strip().split()
        print(predictor.predict(seq))
