#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import yaml
import tqdm
import pickle as pkl

# FOR _csv.Error: field larger than field limit (131072)
import csv

from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import pdb
import gensim
import numpy as np

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.models import TopKDecoder
### Vec mods ###
# from seq2seq.models import VecDecoderRNN
### Done ###
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

from micha_condit import cond_prob

# FOR _csv.Error: field larger than field limit (131072)
csv.field_size_limit(sys.maxsize)

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

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

def initconfig():
    config = {
            'train_path' : './',
            'valid_path' : './',
            'expt_dir' : './',
            'load_model' : 'None'
            }
    return config

if len(sys.argv) > 2:
    _file = sys.argv[2]
    if os.path.isfile(_file):
        with open(_file) as f:
            # with open('config.yml') as f:
            config = yaml.safe_load(f)
    else:
        config = initconfig()
else:
    config = initconfig()




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
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default=config['expt_dir'],
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
parser.add_argument('--device', dest='device_place_holder',
                    help='Device for CUDA: -1 = cpu, 0+ = gpu.\nMultiGPU not supported yet')

opt = parser.parse_args()

# pdb.set_trace()

if 'device' in config:
    if config['device'] == '-1':
        torch.device('cpu')
        DEVICE = 'cpu'
    else:
        torch.device('cuda:' + str(config['device']))
        DEVICE = 'cuda:' + str(config['device'])
elif torch.cuda.is_available():
    torch.device('cuda')
    DEVICE = 'cuda'
else:
    torch.device('cpu')
    DEVICE = 'cpu'

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

# For building the datasets:
max_len = config['max_length']
def len_filter(example):
    # try:
    return len(example.src) <= max_len and len(example.tgt) <= max_len
    # except:
    #     pdb.set_trace()


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
    feats_vocab = checkpoint.feats_vocab
    output_vocab = checkpoint.output_vocab
    # pdb.set_trace()
else:
    # Prepare dataset
    src = SourceField()
    feats = SourceField()
    tgt = TargetField()


    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[('feats', feats), ('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[('feats', feats), ('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )

    src.build_vocab(train, max_size=50000)
    feats.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000)
    input_vocab = src.vocab
    feats_vocab = feats.vocab
    output_vocab = tgt.vocab

    # trying to separate the feats and inputs
    # feats = [x for x in src.vocab.freqs if len(x) > 1]
    # example of getting multihot vector:
    # [1 if x in test_feats else 0 for x in feats]

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(DEVICE, weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = config['encoder embed']
        # TODO is this ideal?
        feat_hidden_size = len(feats.vocab) // 2
        bidirectional = True
        encoder = EncoderRNN(len(src.vocab), feats.vocab,
                             max_len,
                             # TODO can we make these be different sizes?
                             hidden_size,  feat_hidden_size,
                             # hidden_size, hidden_size,
                             bidirectional=bidirectional,
                             rnn_cell='LSTM',
                             variable_lengths=True,
                             n_layers=config['num layers']
                             #,
                             # features=feats
                             )
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
            # aug_size = 0
            aug_size = feat_hidden_size
        # pdb.set_trace()
        decoder = DecoderRNN(len(tgt.vocab),
                             max_len,
                             feat_hidden_size,
                             hidden_size=hidden_size,
                             aug_size=aug_size,
                             dropout_p=float(config['dropout']),
                             input_dropout_p=float(config['dropout']),
                             use_attention=True,
                             bidirectional=bidirectional,
                             rnn_cell='LSTM',
                             eos_id=tgt.eos_id,
                             sos_id=tgt.sos_id,
                             n_layers=config['num layers'])
        # if torch.cuda.is_available():
        #     encoder.cuda()
        #     decoder.cuda()
        # topk_decoder = TopKDecoder(decoder, 3)
        seq2seq = Seq2seq(encoder, decoder)
        # seq2seq = Seq2seq(encoder, topk_decoder)
        if torch.cuda.is_available():
            # pdb.set_trace()
            # seq2seq.to(DEVICE)
            seq2seq.cuda()
            # pdb.set_trace()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        scheduler = StepLR(optimizer.optimizer, 1)
        optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss,
                          batch_size=config['batch size'],
                          checkpoint_every=config['checkpoint_every'],
                          print_every=config['print every'],
                          expt_dir=config['expt_dir'])
                          # expt_dir=opt.expt_dir)

    # TODO add dev eval here for early stopping
    if config['train model']:
        seq2seq = t.train(input_vocab, feats_vocab,
                          seq2seq, train,
                          num_epochs=config['epochs'],
                          vectors=vectors,
                          dev_data=dev,
                          optimizer=optimizer,
                          teacher_forcing_ratio=0.5,
                          resume=opt.resume)

# pdb.set_trace()
predictor = Predictor(seq2seq, input_vocab, feats_vocab, output_vocab, vectors)

# seq2top = Seq2seq(seq2seq.encoder, )
# topk_predictor = Predictor(seq2top, input_vocab, output_vocab, vectors)

if config['pull embeddings']:
    out_vecs = {}


if config['feat embeddings']:
    feats = {}
    of = open(config['feat output'], 'wb')
    # TODO add option to save output
    src = SourceField()
    feat = SourceField()
    tgt = TargetField()
    # pdb.set_trace()
    for key in tqdm(input_vocab.freqs.keys()):
        try:
            guess, enc_out = predictor.predict([key])
        except:
            print("guess, enc_out = predictor.predict([key]) didn't work")
            pdb.set_trace()
        # TODO first try averaging
        # (Pdb)
        # test[3].mean(-1).shape
        # torch.Size([1, 13])
        # (Pdb)
        # test[3].mean(-2).shape
        # torch.Size([1, 600])
        feats[key] = {}
        feats[key]['src'] = key
        feats[key]['tgt'] = key
        feats[key]['guess'] = ''.join(guess)
        feats[key]['embed'] = enc_out
    pkl.dump(feats, of)
    of.close()

if config['eval val']:
    of = open(config['output'], 'w')
    # TODO add option to save output
    correct = 0
    total = 0
    src = SourceField()
    feat = SourceField()
    tgt = TargetField()
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[('feat', feat), ('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    for ex in tqdm(dev.examples):
        # try:
        guess, enc_out, embeddings = predictor.predict(ex.src, ex.feat)
        # except:
        #     print("guess, enc_out = predictor.predict(ex.src, , ex.feat) didn't work")
        #     pdb.set_trace()
        # guess_n, scores = predictor.predict_n(ex.src)
        # pdb.set_trace()
        # topk_guess = topk_predictor.predict(ex.src)
        # [1:] tgt starts with <bos> but model output doesn't
        # normal eval
        if guess == ex.tgt[1:]:
            correct += 1
        # top k
        # for guesses in guess_n:
        #     if guesses == ex.tgt[1:]:
        #         correct += 1
        total += 1
        # normal write out
        srced = ' '.join(ex.src)
        feats = ' '.join(ex.feat)
        tgted = ' '.join(ex.tgt[1:-1])
        guessed = ' '.join(guess[0:-1])
        of.write('\t'.join([srced, tgted, guessed]) + '\n')
        # topk write out
        # for predform, score in zip(guess_n, scores):
        #     of.write('\t'.join([' '.join(ex.src), ' '.join(ex.tgt[1:-1]), ' '.join(predform[0:-1]), str(score)]) + '\n')
        if config['pull embeddings']:
            # TODO first try averaging
            # (Pdb)
            # test[3].mean(-1).shape
            # torch.Size([1, 13])
            # (Pdb)
            # test[3].mean(-2).shape
            # torch.Size([1, 600])
            # embedding = enc_out.mean(-2)
            embedding = np.asarray(embeddings).mean(0)
            # TODO should I look into whether we get different encodings?
            out_vecs[srced] = {}
            out_vecs[srced]['src'] = srced
            out_vecs[srced]['feats'] = feats
            out_vecs[srced]['tgt'] = tgted
            out_vecs[srced]['guess'] = guessed
            out_vecs[srced]['embed'] = embedding

    print("Val accuracy:", correct, "out of", total, correct / total)

if config['pull embeddings']:
    print("Saving embeddings to", config['embeddings file'])
    pkl.dump(out_vecs, open(config['embeddings file'], 'wb'))



# old interactive testing env
if 'ud_out' in config:
    if config['ud_out']:
        outfile = open(config['ud_out_file'], 'w')
        lines = open(config['ud_file'], 'r').readlines()
        for line in tqdm.tqdm(lines):
            line = line.strip().split('\t')
            newline = line
            if len(line) > 6:
                feats = line[5]
                feats = feats.split('|')
                if feats != ['_']:
                    # Needed for japanese:
                    # if feats == ['_']:
                    #     feats = ['None']
                    lemma = line[1]
                    pos = line[3]
                    other_pos = line[4]
                    head_rel = line[7]

                    feats = ' '.join(feats) + ' '
                    feats = pos + ' ' + feats
                    feats = other_pos + ' ' + feats
                    feats = head_rel + ' ' + feats
                    lemma = ' '.join(lemma)

                    seq_str = feats + lemma
                    seq = seq_str.strip().split()
                    out_seq = predictor.predict(seq)[0:-1]
                    outform = ''.join(out_seq)
                    newline[2] = outform
                else:
                    newline[2] = line[1]
            outfile.write('\t'.join(newline) + '\n')



# old interactive testing env
if config['interact']:
    cellIn = "OUT=case=NOM OUT=gen=MAS OUT=num=SG".split(' ')
    cellOut = "OUT=case=NOM OUT=gen=FEM OUT=num=SG".split(' ')
    formIn = list('abgelegen')
    formsOut = [list('katze'), list('abgelene'), list('abgelegene')]
    c = cond_prob()
    probs = c.condPr(formIn, formsOut, cellIn, cellOut, seq2seq, input_vocab, output_vocab)
    print(probs)
    print("Press q to continue to interactive mode.")
    pdb.set_trace()
    while True:
        seq_str = raw_input("Type in a source sequence:")
        seq = seq_str.strip().split()
        print(predictor.predict(seq))
