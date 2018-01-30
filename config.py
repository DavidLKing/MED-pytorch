#!/usr/env/bin python3

class Config:

    def __init__(self):
        self.config = {
            'data' : './data/'
        }

    def getConfig(self):
        self.config['train'] = self.config['data'] + 'train.txt'
        self.config['valid'] = self.config['data'] + 'valid.txt'
        self.config['test'] = self.config['data'] + 'test.txt'

