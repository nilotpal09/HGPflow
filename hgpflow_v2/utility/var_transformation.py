import torch
import numpy as np

class VarTransformation:
    '''
        trans: tranforming the quantities
            eg. x -> log(x), pow(e,m) etc
        scale: scaling the quantities
            eg. x -> (x - mean(x)) / std(x)
        forward: trans + scale
    '''

    def __init__(self, config):
        self.config = config
        self.scale_mode = config['scale_mode']
        self.transformation = config['transformation']



    def trans(self, x):
        if self.transformation == None:
            return x
        if self.transformation == 'pow(x,m)':
            return x ** self.config['m']


    def inv_trans(self, x):
        if self.transformation == None:
            return x
        if self.transformation == 'pow(x,m)':
            return x ** (1 / self.config['m'])



    def scale(self, x):
        if self.scale_mode == None:
            return x
        
        elif self.scale_mode == 'min_max':
            if self.config['range'] == [0,1]:
                return (x - self.config['min']) / (self.config['max'] - self.config['min'])
            elif self.config['range'] == [-1,1]:
                return (x - self.config['min']) / (self.config['max'] - self.config['min']) * 2 - 1
            
            
        elif self.scale_mode == 'standard':
            return (x - self.config['mean']) / self.config['std']
    
    def inv_scale(self, x):
        if self.scale_mode == None:
            return x
        
        elif self.scale_mode == 'min_max':
            if self.config['range'] == [0,1]:
                return x * (self.config['max'] - self.config['min']) + self.config['min']
            elif self.config['range'] == [-1,1]:
                return (x + 1) / 2 * (self.config['max'] - self.config['min']) + self.config['min']
            
        elif self.scale_mode == 'standard':
            return x * self.config['std'] + self.config['mean']



    def forward(self, x):
        x = self.trans(x)
        x = self.scale(x)
        return x
    


    def inverse(self, x):
        x = self.inv_scale(x)
        x = self.inv_trans(x)
        return x

