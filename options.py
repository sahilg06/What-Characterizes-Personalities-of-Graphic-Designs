'''
What Characterizes Personalities of Graphic Designs? SIGGRAPH 2018
Options
'''

import argparse
import os


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--img_rows', type=int, default=300, help='The height of the design')
        self.parser.add_argument('--img_cols', type=int, default=200, help='The width of the design')
        self.parser.add_argument('--img_chns', type=int, default=3, help='The number of channel')
        self.parser.add_argument('--number_of_personalities', type=int, default=15, help='The number of personality label')
        self.parser.add_argument('--feature_output_dim', type=int, default=256,
                                 help='The number of units in the output layer of the design feature network')
        self.parser.add_argument('--feature_dropout_rate', default=[-1, -1, 0.5, -1, 0.5, -1, 0.5, 0.5, 0.5],
                                 help='The dropout rate used in the design feature network')
        self.parser.add_argument('--feature_w_regularizer', type=float, default=0.005,
                                 help='The weight decay used in the design feature network')
        self.parser.add_argument('--batch_norm_flag', type=bool, default=False,
                                 help='Whether use batchnorm in the network')

        self.parser.add_argument('--word_intermediate_dim', type=int, default=[64,64],
                                 help='The number of units in the intermediate layers of the semantic embedding network')
        self.parser.add_argument('--word_w_regularizer', type=float, default=0.005,
                                 help='The weight decay used in the semantic embedding network')

        self.parser.add_argument('--scoring_intermediate_dim', default=[256,128],
                                 help='The number of units in the intermediate layers of the semantic scoring network')
        self.parser.add_argument('--scoring_dropout_rate', default=[0.5, 0.5],
                                 help='The dropout rate used in the semantic scoring network')
        self.parser.add_argument('--scoring_w_regularizer', type=float, default=0.005,
                                 help='The weight decay used in the semantic scoring network')


        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.word_output_dim=self.opt.word_intermediate_dim[-1]
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
