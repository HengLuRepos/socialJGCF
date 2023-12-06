from dataloader import GraphDataset, SocialGraphDataset
douban = SocialGraphDataset("epinions")
douban.getInteractionGraph()