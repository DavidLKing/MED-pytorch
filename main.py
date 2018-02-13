#!/usr/bin/env python3

from io import open
import unicodedata
import string
import re
import random
import argparse
import pdb
import time
import math
import yaml

import torch
import torch.nn as nn
# for batching
import torch.utils.data
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
# for plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

very_verbose = False

with open('config.yml') as f:
    config = yaml.safe_load(f)

# TODO BUG when using torch, we get cuDNN error with sigmorphon data
# TODO when not, we get an index out of range error
# possibly helpful---https://discuss.pytorch.org/t/resolved-torch-backends-cudnn-cudnnerror-4-bcudnn-status-internal-error/4024/3:
# CUDA_CACHE_PATH='/home/david/cudacache' ./main.py -i data/german/train.txt -v data/german/valid.txt -t data/german/test.txt
use_cuda = torch.cuda.is_available()
# use_cuda = False

SOS_token = 0
EOS_token = 1
UNK_token = 2
EPS_token = 3


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        # why does this need to be a super class?
        # oh! 'cause we're getting nn.Module gru stuff
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.tanh(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=config['dropout'], max_length=config['max length']):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<S>", 1: "</S>", 2: "<UNK>", 3: "EPS"}
        self.n_words = 4  # Count <S>, </S>, and <UNK>

    def addSentence(self, sentence):
        # sentence = self.buildSentence(sentence)
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def indices2sent(self, indices):
        return "".join([self.index2word.get(wi, "BUG")
                        for wi in indices])


# # # PLOTTING # # #
class Plot:
    def __init__(self):
        pass

    def showPlot(self, points):
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)


class MED:

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--inputs', help="training file (e.g. data/train.txt", required=True)
        parser.add_argument('-v', '--valid', help="validation file (e.g. data/valid.txt", required=True)
        parser.add_argument('-t', '--test', help="testing file (e.g. data/train.txt", required=True)
        self.args = parser.parse_args()
        # print(self.args)
        self.train = Lang('train')
        self.valid = Lang('valid')
        self.test = Lang('test')

    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def timeSince(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))

    def trainIters(self, encoder, decoder, n_iters, pairs,
                   print_every=1000, plot_every=100,
                   learning_rate=0.01):
        pairs = torch.utils.data.DataLoader(pairs, batch_size=config['batch size'])
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        for batch_num, batch in enumerate(pairs):
            training_pairs = [self.variablesFromPair(self.train, random.choice(batch))
                              for i in range(n_iters)]
            criterion = nn.NLLLoss()

            for iter in range(1, n_iters + 1):
                training_pair = training_pairs[iter - 1]
                input_variable = training_pair[0]
                target_variable = training_pair[1]

                loss = self.train_step(input_variable, target_variable, encoder,
                                       decoder, encoder_optimizer, decoder_optimizer, criterion)
                print_loss_total += loss
                plot_loss_total += loss

                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (self.timeSince(start, iter / n_iters),
                                                 iter, iter / n_iters * 100, print_loss_avg))

                if iter % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

        # TODO Unless we need this, I'm holding off on the moment
        # This application failed to start because it could not find or load the Qt platform plugin "xcb"
        # in "".
        #
        # Available platform plugins are: minimal, offscreen, xcb.
        #
        # Reinstalling the application may fix this problem.
        # Aborted (core dumped)
        # p = plot()
        # p.showplot(plot_losses)

    def train_step(self, input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
                   criterion,
                   max_length=config['max length'], teacher_forcing_ratio=0.5):
        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        use_teacher_forcing = False

        decoder_out = []

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                #  input_word[di]
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_out.append(ni)

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                loss += criterion(decoder_output, target_variable[di])
                if ni == EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        if very_verbose:
            print(self.train.indices2sent(decoder_out))

        return loss.data[0] / target_length

    # # # EVALUATION # # #
    def evaluate(self, encoder, decoder, lang, sentence, max_length=config['max length']):
        # ORIGINALLY THIS WAS input_lang
        input_variable = self.variableFromSentence(lang, sentence)
        input_length = input_variable.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            # decoder_input, decoder_hidden, encoder_outputs)
            # decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                # ORIGINALLY THIS WAS output_lang
                # decoded_words.append(output_lang.index2word[ni])
                decoded_words.append(lang.index2word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        return decoded_words, decoder_attentions[:di + 1]

    def manualEval(self, pairs, lang, encoder, decoder):
        correct = 0
        total = 0
        for pair in pairs:
            total += 1
            guess = self.evaluate(encoder, decoder, lang, pair[0], max_length=config['max length'])
            if ' '.join(guess[0]) == pair[1] + ' <EOS>':
                correct += 1
        print("accuracy:", correct / total)

    def evaluateRandomly(self, encoder, pairs, lang, decoder, n=10):
        for i in range(n):
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = self.evaluate(encoder, decoder, lang, pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

    # # # DENSE VECTORS # # #
    def indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def variableFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        result = Variable(torch.LongTensor(indexes).view(-1, 1))
        if use_cuda:
            return result.cuda()
        else:
            return result

    def variablesFromPair(self, lang, pair):
        input_variable = self.variableFromSentence(lang, pair[0])
        target_variable = self.variableFromSentence(lang, pair[1])
        return (input_variable, target_variable)

    # # # DATA PREP # # #

    def splitdata(self, _file):
        lines = open(_file, 'r').readlines()
        inseq = []
        outseq = []
        for l in lines:
            l = l.split('\t')
            new_l = ' '.join(l[0])
            feats = [f for f in l[1].split(',')]
            new_l += ' ' + ' '.join(feats)
            inseq.append(new_l)
            outseq.append(' '.join(l[2].strip()))
        return inseq, outseq

    def pairdata(self, seqin, seqout, dataset):
        pairseq = [[s1, s2] for s1, s2 in zip(seqin, seqout)]
        for s1, s2 in zip(seqin, seqout):
            dataset.addSentence(s1)
            dataset.addSentence(s2)
        return pairseq

    # # # MAIN FUNCTION # # #

    def main(self):
        train_in, train_out = self.splitdata(self.args.inputs)
        valid_in, valid_out = self.splitdata(self.args.valid)
        test_in, test_out = self.splitdata(self.args.test)
        train = self.pairdata(train_in, train_out, self.train)
        valid = self.pairdata(valid_in, valid_out, self.valid)
        test = self.pairdata(test_in, test_out, self.test)
        print(random.choice(train))
        # possibly related to the number of embeddings: https://github.com/pytorch/pytorch/issues/1998
        en = EncoderRNN(config['encoder embed'], self.train.n_words)
        de = DecoderRNN(self.train.n_words, config['decoder embed'])
        if use_cuda:
            en = en.cuda()
            de = de.cuda()
        # TODO only run this during training---iters = num epochs * lines / batches
        # len(train) gets us that
        # for batch_num, batch in enumerate(train):
        # TODO 50 is the current hard coded batch size---make that an option
        train_iter = config['epochs'] * len(train)
        if config['train']:
            self.trainIters(en, de, train_iter, train, print_every=config['print every'],
                            plot_every=config['plot every'], learning_rate=config['learning rate'])
        # TODO have guesses  written out so we can do error analysis, sig testing, etc... later on
        if config['eval val']:
            print("Evaluating the validation set:")
            self.manualEval(valid, self.valid, en, de)
        if config['eval test']:
            print("Evaluating the test set:")
            self.manualEval(test, self.test, en, de)
        # pdb.set_trace()


if __name__ == '__main__':
    m = MED()
    m.main()
