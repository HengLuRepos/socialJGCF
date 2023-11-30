from pprint import pprint
import time

import dataloader
import model
import world


if world.dataset in ['lastfm', 'ciao', 'epinions', 'douban', 'gowalla']:
    if world.model_name in ['SocialLGN','SocialSimGCL']:
        dataset = dataloader.SocialGraphDataset(world.dataset)
    elif world.model_name in ['LightGCN', 'SimGCL']:
        dataset = dataloader.GraphDataset(world.dataset)
    elif world.model_name in ['bpr']:
        dataset = dataloader.PairDataset(world.dataset)

print('===========config================')
pprint(world.config)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'bpr': model.PureBPR,
    'LightGCN': model.LightGCN,
    'SimGCL': model.SimGCL,
    'SocialLGN': model.SocialLGN,
    'SocialSimGCL': model.SocialSimGCL,
    'JGCF': model.JGCF,
    'SocialJGCF': model.SocialJGCF,
}
