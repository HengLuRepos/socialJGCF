import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # for hyper parameters
    parser.add_argument('-m', '--model', type=str, default='SocialLGN')
    parser.add_argument('-d', '--dataset', type=str, default='lastfm')
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalization")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--epochs', type=int, default=10000)
    # for deep model
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of graphs")
    # normally unchanged
    parser.add_argument('--topks', nargs='?', default="[10, 20]",
                        help="@k test list")
    parser.add_argument('--testbatch', type=str, default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--noise-norm', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=0.2, help="temperature")
    parser.add_argument('--lam', type=float, default=0.5, help='cl_loss')
    parser.add_argument('--a', type=float, default=1., help='a of Jacobian')
    parser.add_argument('--b', type=float, default=1., help='b of Jacobian')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    return parser.parse_args()
