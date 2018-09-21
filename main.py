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
# from torch.autograd import Variable
import torch.nn
from torch import optim
import torch.nn.functional as F

# for plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pickle as pkl

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
    def __init__(self, input_size, hidden_size, dropout_p, variable_lengths=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
        # self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.rnn = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, input_var, hidden, mask, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
        input_var (batch, seq_len): tensor containing the features of the input sequence.type
        input_lengths (list of int, optional): A list that contains the lengths of sequences
        in the mini-batch
        Returns: output, hidden
        - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        # test = torch.nn.utils.rnn.PackedSequence(input_var, config['batch size'])
        # input_var = input_var * mask
        embedded = self.embedding(input_var)
        # Variable(mask.unsqueeze(-1)) * embedded
        # mask.unsqueeze(-1)) * embedded
        # convert mask to mask size
        mask = mask.unsqueeze(2)
        mask = mask.repeat(1, 1, self.hidden_size)
        # TODO is this the correct place to apply the mask
        embedded = embedded * mask
        # embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    # def forward(self, input, hidden):
    #     embedded = self.embedding(input).view(1, 1, -1)
    #     embedded = self.input_dropout(embedded)
    #     output = embedded
    #     output, hidden = self.gru(output, hidden)
    #     return output, hidden

    def initHidden(self, size):
        result = torch.zeros(2, size, self.hidden_size)
        # result = Variable(torch.zeros(1, 1, self.hidden_size))
        # result = torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, dropout_p=config['dropout'], max_length=config['max length']):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size * 2
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # TODO WHY IS this over the max length? Possibly wrong from tutorial
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, mask):
        # DLK---I don't think this is required
        # embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.embedding(input)
        embedded = embedded.squeeze(1).unsqueeze(0)
        mask = mask.squeeze(1).unsqueeze(0)
        embedded = self.dropout(embedded)
        '''
        # atth_weights is coming in at 10, 50, 1)
        # Is this correct?
        attn_weights = F.softmax(
                                self.attn(torch.cat(
                                    (embedded, hidden), -1)
                                    ), dim=2
                                )
        '''
        mask = mask.unsqueeze(2)
        mask = mask.repeat(1, 1, self.hidden_size)
        embedded = embedded * mask
        '''
        # torch.bmm = (b, n, m) @ (b, m, p) -> (b, n, p)
        attn_weights = attn_weights.squeeze(0).unsqueeze(-1)
        encoder_outputs = encoder_outputs.unsqueeze(1)
        pdb.set_trace()
        attn_applied = torch.bmm(attn_weights,
                                 encoder_outputs)
        emb = embedded.squeeze(0).unsqueeze(1).repeat(1, 50, 1)
        output = torch.cat((emb, attn_applied), -1)
        pdb.set_trace()
        output = self.attn_combine(output).view(config['batch size'], -1)
        # output = F.relu(output)
        '''
        embedded = embedded.squeeze(0).unsqueeze(1)
        output = F.relu(embedded)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output), dim=2)
        # return output, hidden, attn_weights
        return output, hidden, hidden

    def initHidden(self):
        result = torch.zeros(1, 1, self.hidden_size)
        if use_cuda:
            return result.cuda()
        else:
            return result


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<S>": 0, "</S>": 1, "<UNK>": 2, "<EPS>": 3}
        self.word2count = {}
        self.index2word = {0: "<S>", 1: "</S>", 2: "<UNK>", 3: "<EPS>"}
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
        # print(self.args)
        # Initialize language objects
        self.train = Lang('train')
        self.valid = Lang('valid')
        self.test = Lang('test')

    def getArgs(self):
        parser = argparse.ArgumentParser()
        self.args = parser.parse_args()

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

    # # # PADDING FOR MINIBATCHING # # #

    def pad(self, seq, lang):
        # max length
        longest = max([x.size()[0] for x in seq])
        # hack to make padding work for the longest seq
        longest += 1
        if use_cuda:
            pad = lambda x: torch.cat((
                x.squeeze(-1),
                torch.cuda.LongTensor(
                    [lang.word2index['</S>']] * (longest - x.size()[0])
                )
            ))
        else:
            pad = lambda x: torch.cat((
                x.squeeze(-1),
                torch.LongTensor(
                    [lang.word2index['</S>']] * (longest - x.size()[0])
                )
            ))
        newseq = torch.stack([pad(s) for s in seq])
        mask = torch.stack([self.build_mask(x, longest) for x in seq])
        if use_cuda:
            mask = mask.cuda()
        return newseq, mask

    def build_mask(self, seq, longest):
        ones = torch.ones(len(seq))
        zeros = torch.zeros(longest- len(seq))
        return torch.cat((ones, zeros))

    def trainIters(self, encoder, decoder, epochs, n_iters, in_pairs, valid, test,
                   print_every=config['print every'], plot_every=config['plot every'],
                   learning_rate=config['learning rate']):
        # Record keeping #
        epoch = 0
        train_len = len(in_pairs)
        # epoch_num = train_len // config['batch size']
        place = 0

        # batching the data
        # pairs = in_pairs
        pairs = torch.utils.data.DataLoader(in_pairs, batch_size=config['batch size'])
        # record keeping cont.
        epoch_num = len(pairs)
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.Adadelta(encoder.parameters(), lr=learning_rate)
        # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adadelta(decoder.parameters(), lr=learning_rate)
        # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

        for num in range(epochs):
            epoch = num
            for batch_num, batch in enumerate(pairs):

                training_pairs = [self.variablesFromPair(self.train, (batch[0][i], batch[1][i]))
                                  for i in range(len(batch[0]))]
                # inputs = [self.variableFromSentence(self.train, batch[0][i]) for i in len(batch[0])]
                # outputs = [self.variableFromSentence(self.train, batch[1][i]) for i in len(batch[0])]
                # training_pairs = [self.variablesFromPair(self.train, random.choice(pairs))
                # for i in range(n_iters)]/

                place += config['batch size']

                # criterion = nn.NLLLoss()
                # criterion = nn.MSELoss()
                criterion = nn.CrossEntropyLoss()

                # TODO MAKE SURE MINITBATCHING IS ACTUALLY MAKING MINIBATCHES
                # TODO should self.train be 'lang' here?
                # for iter in range(1, len(batch[0]) + 1):
                # training_pair = training_pairs [iter - 1]
                input_variable = [x[0] for x in training_pairs]
                # input_variable = torch.stack(self.pad([x[0] for x in training_pairs], self.train))
                # input_variable = training_pair[0]
                target_variable = [y[1] for y in training_pairs]
                # target_variable = torch.stack(self.pad([y[1] for y in training_pairs], self.train))
                # target_variable = training_pair[1]
                # input_variable = inputs[iter - 1]
                # target_variable = outputs[iter - 1]

                if len(input_variable) == config['batch size']:
                    loss = self.train_step(input_variable, target_variable, encoder,
                                           decoder, encoder_optimizer, decoder_optimizer, criterion)
                    print_loss_total += loss
                    plot_loss_total += loss

                    if place % print_every == 0:
                        # if True:
                        print_loss_avg = print_loss_total / print_every
                        print_loss_total = 0
                        print('%s (%d %d%%) %.4f' % (self.timeSince(start, place / n_iters),
                                                     place, place / n_iters * 100, print_loss_avg))

                    # SAMPLING for stdout monitoring
                    if place % config['print sample every'] == 0 and place > 500:
                        # if True:
                        """
                        TODO the iter > 500 thing is a hack. Do we need to be concerned that
                        the decoder predicts OOV character in the beginning of training?
                        """
                        sample = random.choice(in_pairs)
                        sample_in = sample[0]
                        sample_targ = sample[1]
                        guess = self.evaluate(encoder, decoder, self.train, sample_in, max_length=config['max length'])
                        print(" Input:", sample_in, "\n",
                              "Target:", sample_targ, "\n",
                              "Predicted:", ''.join(guess[0])
                              )

                    if place % plot_every == 0:
                        plot_loss_avg = plot_loss_total / plot_every
                        plot_losses.append(plot_loss_avg)
                        plot_loss_total = 0
            
            print("### FINISHED EPOCH", epoch + 1, "of", epochs, "###")
            # TODO can we abstract this into a function?
            if config['eval val']:
                print("Evaluating the validation set:")
                self.manualEval(valid, self.valid, encoder, decoder)
            if config['eval test']:
                print("Evaluating the test set:")
                self.manualEval(test, self.test, encoder, decoder)
            # if epoch > 20:
            #     print("Hit 'c' to continue, and 'q' to quit")

        # TODO make plotting into a switch
        # p = Plot()
        # p.showPlot(plot_losses)

    def train_step(self, input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
                   criterion,
                   max_length=config['max length'], teacher_forcing_ratio=0.5):
        encoder_hidden = encoder.initHidden(config['batch size'])

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_variable, input_mask = self.pad(input_variable, self.train)
        # input_variable = torch.stack(input_variable).squeeze(-1)
        # input_mask = input_mask.unsqueeze(-1)
        target_variable, target_mask = self.pad(target_variable, self.train)
        # target_variable = torch.stack(target_variable).squeeze(-1)
        # target_mask = target_mask.unsqueeze(-1)
        # TODO I believe Transposing is the correct things to do
        # input_variable = input_variable.t()
        # target_variable = target_variable.t()

        # input_length = input_variable.shape[1]
        # target_length = target_variable.shape[1]
        # input_length = input_variable.shape[1]
        # target_length = target_variable.shape[1]

        # stuff I'm trying:, the internet keeps saying this is the way, but I don't believe it
        # seq_len = [x.shape[0] for x in input_variable]
        # pack = torch.nn.utils.rnn.pack_padded_sequence(self.pad(input_variable, self.train), seq_len, batch_first=True)

        # added *2 for bidirectional input
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size * 2)
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        loss = 0


        # THIS IS WORKING ON SEQUENCES, FINALLY, BUT IT IS NOT WORKING ON MINIBATCHES
        # I THOUGHT THIS WAS A DIM ISSUE, BUT STACKING THE ENCODER_HIDDEN DOESN'T SEEM TO WORK
        # encoder(input_variable, torch.stack(encoder_hidden.repeat(50, 1, 1)))
        # ** *AssertionError
        # encoder(input_variable[0], encoder_hidden)
        # THE ABOVE WORKS
        encoder_output, encoder_hidden = encoder(input_variable, encoder_hidden, input_mask)

        # TODO I'm pretty sure this is correct. Enocoder output is batch x length x dim
        # decoder needs time step x dim iteratively over the length
        encoder_outputs = encoder_output.permute(1, 0, 2)
        # encoder_output, encoder_hidden = encoder(input_variable, encoder_hidden)
        # encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = torch.LongTensor([[SOS_token]])
        # DLK lengthening for batch
        decoder_input = decoder_input.repeat(config['batch size'],1)
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        use_teacher_forcing = False

        decoder_out = []

        if use_teacher_forcing:
            # TODO be ready to fix this
            # Teacher forcing: Feed the target as the next input
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, target_mask)
            loss += criterion(decoder_output, target_variable)
            decoder_input = target_variable  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            # TODO finish this
            # waiting on Lifeng to verify how this works
            decoder_out = []
            # TODO what to do about training data that is too small for a batch
            decoder_hidden = decoder_hidden.view(1,config['batch size'], -1)
            # decoder_hidden = decoder_hidden.view(1, batch_len, -1)
            for di in range(target_variable.size()[1]):
                mask = target_mask.t()[di].unsqueeze(-1)
                # TODO delete this once loss is figured out
                # decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_output, decoder_hidden, attn_weights = decoder(
                        decoder_input, decoder_hidden, encoder_outputs[di], mask)
                decoder_out.append(decoder_output)
                # TODO here's where beam search and n-best come from
                topv, topi = decoder_output.data.topk(1)
                ni = topi.squeeze(-1)

                decoder_out.append(ni)

                decoder_input = ni
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                
                # TROUBLESHOOTING
                # print("ni", ni)
                # print("di", di)
                # print("target var", target_variable.t()[di])
                # print('same?', ni.squeeze(1) == target_variable.t()[di])
                # pdb.set_trace()

                loss += criterion(decoder_output.squeeze(1), target_variable.t()[di])
                # loss += criterion(decoder_output, target_variable[di])
            # if ni == EOS_token:
                # break

        # TODO touble check loss
        # Python 3.5.2 (default, Nov 23 2017, 16:37:01)
        # [GCC 5.4.0 20160609] on linux
        # Type "help", "copyright", "credits" or "license" for more information.
        # >>> import math
        # >>> math.log(1/50)
        # -3.912023005428146
        # >>> math.log(1/2)
        # -0.6931471805599453
        # >>> math.exp(-.0004)
        # 0.9996000799893344
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        if very_verbose:
            print(self.train.indices2sent(decoder_out))

        # return loss.data[0] * config['batch size']
        return loss.data[0]

    # # # EVALUATION # # #
    def evaluate(self, encoder, decoder, lang, sentence, max_length=config['max length']):
        # ORIGINALLY THIS WAS input_lang
        input_variable = self.variableFromSentence(lang, sentence)
        input_length = input_variable.size()[0]
        input_mask = torch.ones(input_length).unsqueeze(-1)
        input_mask = input_mask.cuda() if use_cuda else input_mask

        encoder_hidden = encoder.initHidden(1)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size * 2)
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        # TODO getting weird dimensionalities.
        # input_variable.t() seems to help, but isn't the whole answer
        encoder_output, encoder_hidden = encoder(input_variable.t(), encoder_hidden, input_mask.t())
        encoder_outputs = encoder_output

        decoder_input = torch.LongTensor([[SOS_token]])  # SOS
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden
        # decoder_hidden = decoder_hidden.view(1,config['batch size'], -1)
        decoder_hidden = decoder_hidden.view(1, 1, -1)
       
        decoded_words = []
        decoder_out = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            target_mask = torch.ones(1).unsqueeze(-1)
            target_mask = target_mask.cuda() if use_cuda else target_mask
            # TODO delete this once loss is figured out
            # decoder_output, decoder_hidden, decoder_attention = decoder(
            try:
                # TODO encoder_outputs[0][di] mucking things up
                decoder_output, decoder_hidden, fake_attn = decoder(
                    decoder_input, decoder_hidden, encoder_outputs[0][0], target_mask)
                    # decoder_input, decoder_hidden, encoder_outputs[0][di], target_mask)
            except:
                pdb.set_trace()
            decoder_out.append(decoder_output)
            # TODO here's where beam search and n-best come from
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(-1)

            decoder_out.append(ni)
            
            decoder_input = ni
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            if ni[0][0] == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                # ORIGINALLY THIS WAS output_lang
                # decoded_words.append(output_lang.index2word[ni])
                try:
                    decoded_words.append(lang.index2word[int(ni[0][0])])
                except:
                    pdb.set_trace()


            # decoder_input = torch.LongTensor([[ni]]))
            # decoder_input = decoder_input.cuda() if use_cuda else decoder_input

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
        result = torch.LongTensor(indexes).view(-1, 1)
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
            # new_l += ' ' + ' '.join(feats)
            # testing
            new_l = ' '.join(feats) + ' ' + new_l
            inseq.append(new_l)
            outseq.append(' '.join(l[2].strip()))
        return inseq, outseq

    def pairdata(self, seqin, seqout, dataset):
        pairseq = [[s1, s2] for s1, s2 in zip(seqin, seqout)]
        for s1, s2 in zip(seqin, seqout):
            dataset.addSentence(s1)
            dataset.addSentence(s2)
        return pairseq

    # # # MODEL SAVING # # #

    def saveModel(self, encoder, decoder, tr_lang, va_lang, te_lang):
        model = open(config['save name'], 'wb')
        pkl.dump((encoder, decoder, tr_lang, va_lang, te_lang), model)
        model.close()

    def loadModel(self):
        # encoder, decoder, tr_lang, va_lang, te_lang
        with open(config['save name'], 'rb') as model:
            return pkl.load(model)

    # # # MAIN FUNCTION # # #

    def main(self):
        self.getArgs()
        # train_in, train_out = self.splitdata(self.args.inputs)
        train_in, train_out = self.splitdata(config['train'])
        # valid_in, valid_out = self.splitdata(self.args.valid)
        valid_in, valid_out = self.splitdata(config['valid'])
        # test_in, test_out = self.splitdata(self.args.test)
        test_in, test_out = self.splitdata(config['test'])
        train = self.pairdata(train_in, train_out, self.train)
        valid = self.pairdata(valid_in, valid_out, self.valid)
        test = self.pairdata(test_in, test_out, self.test)
        print("Sample training:\n", random.choice(train))
        if config['load model']:
            en, de, self.train, self.valid, self.text = self.loadModel()
        else:
            # possibly related to the number of embeddings: https://github.com/pytorch/pytorch/issues/1998
            en = EncoderRNN(self.train.n_words, config['encoder embed'], dropout_p=config['dropout'], variable_lengths=True)
            # EnboderRNN(config['encoder embed'], self.train.n_words)
            # de = DecoderRNN(self.train.n_words, config['decoder embed'])
            de = AttnDecoderRNN(self.train.n_words, config['decoder embed'], dropout_p=config['dropout'])
            # de = DecoderRNN(self.train.n_words, config['decoder embed'] * 2)
            if use_cuda:
                en = en.cuda()
                de = de.cuda()
        # TODO only run this during training---iters = num epochs * lines / batches
        # len(train) gets us that
        train_iter = (config['epochs'] * len(train)) // config['batch size']
        if config['train']:
            self.trainIters(en, de, config['epochs'], train_iter, train, valid, test,
                            print_every=config['print every'],
                            plot_every=config['plot every'], 
                            learning_rate=config['learning rate'])
        # TODO have guesses  written out so we can do error analysis, sig testing, etc... later on
        if config['eval val']:
            print("Evaluating the validation set:")
            self.manualEval(valid, self.valid, en, de)
        if config['eval test']:
            print("Evaluating the test set:")
            self.manualEval(test, self.test, en, de)
        if config['save model']:
            print("Saving model to:", config['save name'])
            self.saveModel(en, de, self.train, self.valid, self.test)


if __name__ == '__main__':
    m = MED()
    m.main()
