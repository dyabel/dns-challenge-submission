from torch import nn
import sys
sys.path.append("..")
from wavenet.tcn import TemporalConvNet
import wavenet.sequnet as sequnet
import wavenet.sequnet_res as sequnet_res
import argparse
import numpy as np
from s4.s4_model import S4Layer

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

# parser = argparse.ArgumentParser(description='Sequence Modeling - Word-level Language Modeling')

# parser.add_argument('--batch_size', type=int, default=16, metavar='N',
#                     help='batch size (default: 16)')
# parser.add_argument('--cuda', action='store_true',
#                     help='use CUDA (default: False)')
# parser.add_argument('--dropout', type=float, default=0.5,
#                     help='dropout applied to layers (default: 0.5)')
# parser.add_argument('--emb_dropout', type=float, default=0.25,
#                     help='dropout applied to the embedded layer (default: 0.25)')
# parser.add_argument('--clip', type=float, default=0.4,
#                     help='gradient clip, -1 means no clip (default: 0.4)')
# parser.add_argument('--epochs', type=int, default=100,
#                     help='upper epoch limit (default: 100)')
# parser.add_argument('--ksize', type=int, default=3,
#                     help='kernel size (default: 3)')
# parser.add_argument('--data', type=str, default='./data/penn',
#                     help='location of the data corpus (default: ./data/penn)')
# parser.add_argument('--emsize', type=int, default=600,
#                     help='size of word embeddings (default: 600)')
# parser.add_argument('--levels', type=int, default=4,
#                     help='# of levels (default: 4)')
# parser.add_argument('--log_interval', type=int, default=100, metavar='N',
#                     help='report interval (default: 100)')
# parser.add_argument('--lr', type=float, default=1e-3,
#                     help='initial learning rate (default: 1e-3)')
# parser.add_argument('--nhid', type=int, default=600,
#                     help='number of hidden units per layer (default: 600)')
# parser.add_argument('--seed', type=int, default=1111,
#                     help='random seed (default: 1111)')
# parser.add_argument('--tied', action='store_false',
#                     help='tie the encoder-decoder weights (default: True)')
# parser.add_argument('--optim', type=str, default='Adam',
#                     help='optimizer type (default: SGD)')
# parser.add_argument('--validseqlen', type=int, default=45,
#                     help='valid sequence length (default: 45)')
# parser.add_argument('--seq_len', type=int, default=117,
#                     help='total sequence length, including effective history (default: 117)')
# parser.add_argument('--corpus', action='store_true',
#                     help='force re-make the corpus (default: False)')
# parser.add_argument('--model', type=str, default='tcn',
#                     help='Model to use (tcn/sequnet)')
# parser.add_argument('--hyper_iter', type=int, default=20,
#                     help='Hyper-parameter optimisation runs (default: 20)')
# parser.add_argument('--mode', type=str, default='hyper',
#                     help='Training mode - hyper: Perform hyper-parameter optimisation, return best configuration. '
#                          'train: Train single model with given parameters')
# parser.add_argument('--experiment_name', type=str, default=str(np.random.randint(0, 100000)),
#                     help="Optional name of experiment, used for saving model checkpoint")

# args = parser.parse_args()


class TCN(nn.Module):

    def __init__(self, input_size):
        super(TCN, self).__init__()
        args = dotdict(dict(batch_size=16, clip=0.4, corpus=False, cuda=False, data='./data/penn', 
                    dropout=0.5, emb_dropout=0.25, emsize=600, epochs=100, experiment_name='74256', 
                    hyper_iter=20, ksize=3, levels=4, log_interval=100, lr=0.001, 
                    mode='hyper', model='tcn', nhid=600, optim='Adam', seed=1111, 
                    seq_len=117, tied=True, validseqlen=45))
        self.encoder = nn.Linear(input_size, args.emsize)
        num_channels = [args.nhid] * (args.levels - 1) + [args.emsize]

        if args.model == "tcn":
            self.conv = TemporalConvNet(args.emsize, num_channels, kernel_size=args.ksize, dropout=args.dropout)
        elif args.model == "sequnet" or args.model == "sequnet_res":
            if args.model == "sequnet":
                self.conv = sequnet.Sequnet(args.emsize, num_channels, args.emsize, kernel_size=args.ksize,
                                             dropout=args.dropout, target_output_size=args.validseqlen)
            else:
                self.conv = sequnet_res.Sequnet(args.emsize, num_channels[0], len(num_channels), args.emsize,
                                                kernel_size=args.ksize, target_output_size=args.validseqlen)
            args.validseqlen = self.conv.output_size
            args.seq_len = self.conv.input_size
            print("Using Seq-U-Net with " + str(args.validseqlen) + " outputs and " + str(args.seq_len) + " inputs")
        else:
            raise NotImplementedError("Could not find this model " + args.model)

        self.s4 =  S4Layer(d_input=600,
                        d_output=600,
                        d_model=600,
                        n_layers=4,
                        dropout=0.2,
                        prenorm=False,
                        args=args) # Build Neural Network
        self.decoder = nn.Linear(num_channels[-1], input_size)
        # if args.tied:
        #     if num_channels[-1] != args.emsize:
        #         raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder.weight = self.encoder.weight
        #     print("Weight tied")
        self.drop = nn.Dropout(args.emb_dropout)
        self.emb_dropout = args.emb_dropout
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.drop(self.encoder(input))
        y = self.conv(emb.transpose(1, 2)).transpose(1, 2)
        # print(y.shape, self.decoder.weight.shape, self.decoder.bias.shape)
        y = self.s4(y)
        y = self.decoder(y)
        return y.contiguous()