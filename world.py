import os
from os.path import join
from warnings import simplefilter

import torch

from parse import parse_args

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "./"
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(ROOT_PATH, 'runs')
FILE_PATH = join(ROOT_PATH, 'checkpoints')

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

config = {}
all_dataset = ['lastfm', 'ciao', 'douban', 'epinions']
all_models = ['bpr', 'LightGCN', 'SocialLGN','SimGCL','SocialSimGCL', 'JGCF', 'SocialJGCF']

config['layer'] = args.layer

config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim

config['lr'] = args.lr
config['decay'] = args.decay

config['test_u_batch_size'] = args.testbatch

config['noise_norm'] = args.noise_norm
config['lam'] = args.lam
config['tau'] = args.tau
config['a'] = args.a
config['b'] = args.b
config['alpha'] = args.alpha

GPU = torch.cuda.is_available()
device = torch.device('cuda:1' if GPU else "cpu")
seed = args.seed
LOAD = args.load
PATH = './checkpoints'

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

TRAIN_epochs = args.epochs
topks = eval(args.topks)

simplefilter(action="ignore", category=FutureWarning)
