import world
import torch
import torch.nn as nn
import torch.nn.functional as F

class PureBPR(nn.Module):
    def __init__(self, config, dataset):
        super(PureBPR, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        print("using Normal distribution initializer")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss


class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self._init_weight()

    def _init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['layer']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.interactionGraph = self.dataset.getInteractionGraph()
        print(f"{world.model_name} is already to go")

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        G = self.interactionGraph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(G, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.final_user, self.final_item
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss
class JGCF(LightGCN):
    def __init__(self, config, dataset):
        super(JGCF, self).__init__(config, dataset)
        self.a = config['a']
        self.b = config['b']
        self.alpha = config['alpha']
    def computer(self):
        all_num = self.num_users + self.num_items
        eye = torch.sparse_coo_tensor([range(all_num),range(all_num)],torch.ones(all_num), dtype=torch.float32, device=self.interactionGraph.device)
        user_weight = self.embedding_user.weight
        item_weight = self.embedding_item.weight
        embed = torch.cat([user_weight, item_weight])

        p0 = eye
        p1 = (self.a - self.b)/2 * eye + (self.a + self.b)/2 * self.interactionGraph
        embs = [torch.sparse.mm(eye, embed), torch.sparse.mm(p1, embed)]
        for k in range(2, self.n_layers + 1):
            theta_1 = (2*k + self.a + self.b) * (2*k + self.a + self.b - 1) / ((k + self.a + self.b) * 2*k)
            theta_2 = ((2*k + self.a + self.b - 1)*(self.a**2 - self.b**2)) /((2*k + self.a + self.b - 2) * (k + self.a + self.b) * 2 * k)
            emb_k = theta_1 * torch.sparse.mm(self.interactionGraph, embs[-1]) + theta_2 * embs[-1]
            theta_3 = ((k + self.a - 1) * (k + self.b - 1) * (2*k + self.a + self.b)) / (k*(self.a + self.b + k)*(2*k + self.a + self.b -2))
            emb_k -= theta_3 * embs[-2]
            embs.append(emb_k)
        band_stop = torch.stack(embs, dim=1).mean(dim=1)
        band_pass = torch.tanh(self.alpha * embed - band_stop)
        out = torch.hstack([band_stop, band_pass])
        users, items = torch.split(out, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return users, items

class SimGCL(LightGCN):
    def __init__(self, config, dataset):
        super(SimGCL, self).__init__(config, dataset)
        self.noise_norm = config['noise_norm']
        self.lam = config['lam']
        self.tau = config['tau']
    def cl_loss(self, user_idx, item_idx):
        device = self.embedding_item.weight.device
        user_idx = torch.unique(torch.tensor(user_idx, device=device))
        item_idx = torch.unique(torch.tensor(item_idx, device=device))
        users_1, items_1 = self.computer(perturbed=True)
        users_2, items_2 = self.computer(perturbed=True)
        user_cl_loss = self.infoNCE(users_1[user_idx], users_2[user_idx])
        item_cl_loss = self.infoNCE(items_1[item_idx], items_2[item_idx])
        return self.lam * (user_cl_loss + item_cl_loss)

    def random_noise(self, tensor):
        noise = torch.rand_like(tensor, device=tensor.device)
        noise = F.normalize(noise, dim=1) * self.noise_norm * torch.sign(tensor)
        return noise

    def computer(self, perturbed=False):
        user_weight = self.embedding_user.weight
        item_weight = self.embedding_item.weight
        embed = torch.cat([user_weight, item_weight])
        embs = []
        for layer in range(self.n_layers):
            embed = torch.sparse.mm(self.interactionGraph, embed)
            if perturbed:
                embed = embed + self.random_noise(embed)
            embs.append(embed)
        embs = torch.stack(embs, dim=1)
        out = torch.mean(embs, dim=1)
        users, items = torch.split(out, [self.num_users, self.num_items])
        #if not perturbed:
        #    self.final_user, self.final_item = users, items
        return users, items
    def infoNCE(self, emb1, emb2):
        emb1, emb2 = F.normalize(emb1, dim=1), F.normalize(emb2, dim=1)
        pos_score = (emb1 @ emb2.T) / self.tau
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()
class SocialJGCF(JGCF):
    def _init_weight(self):
        super(SocialJGCF, self)._init_weight()
        self.socialGraph = self.dataset.getSocialGraph()
        self.Graph_Comb = Graph_Comb_JGCF(self.latent_dim)
    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        A = self.interactionGraph
        S = self.socialGraph
        all_num = self.num_users + self.num_items
        inter_eye = torch.sparse_coo_tensor([range(all_num),range(all_num)],torch.ones(all_num), dtype=torch.float32, device=A.device)
        social_eye = torch.sparse_coo_tensor([range(self.num_users),range(self.num_users)],torch.ones(self.num_users), dtype=torch.float32, device=S.device)
        
        p0_inter = inter_eye
        p1_inter = (self.a - self.b)/2 * inter_eye + (self.a + self.b)/2 * A
        
        p0_social = social_eye
        p1_social = (self.a - self.b)/2 * social_eye + (self.a + self.b)/2 * S

        embs = [torch.sparse.mm(inter_eye, all_emb), torch.sparse.mm(p1_inter, all_emb)]
        social_embs = [torch.sparse.mm(social_eye, users_emb), torch.sparse.mm(p1_social, users_emb)]
        for k in range(2, self.n_layers + 1):
            theta_1 = (2*k + self.a + self.b) * (2*k + self.a + self.b - 1) / ((k + self.a + self.b) * 2*k)
            theta_2 = ((2*k + self.a + self.b - 1)*(self.a**2 - self.b**2)) /((2*k + self.a + self.b - 2) * (k + self.a + self.b) * 2 * k)
            theta_3 = ((k + self.a - 1) * (k + self.b - 1) * (2*k + self.a + self.b)) / (k*(self.a + self.b + k)*(2*k + self.a + self.b -2))
            inter_emb_k = theta_1 * torch.sparse.mm(A, embs[-1]) + theta_2 * embs[-1]
            inter_emb_k -= theta_3 * embs[-2]
            embs.append(inter_emb_k)

            social_emb_k = theta_1 * torch.sparse.mm(S, social_embs[-1]) + theta_2 * social_embs[-1]
            social_emb_k -= theta_3 * social_emb_k[-2]
            social_embs.append(social_emb_k)
        band_stop_inter = torch.stack(embs, dim=1).mean(dim=1)
        band_pass_inter = torch.tanh(self.alpha * all_emb - band_stop_inter)
        inter_out = torch.hstack([band_stop_inter, band_pass_inter])
        inter_users_emb, items_emb_next = torch.split(inter_out, [self.num_users, self.num_items])
        
        band_stop_social = torch.stack(social_embs, dim=1).mean(dim=1)
        band_pass_social = torch.tanh(self.alpha * users_emb - band_stop_social)
        users_social = torch.hstack([band_stop_social, band_pass_social])
        users_emb_next = self.Graph_Comb(users_social, inter_users_emb)
        self.final_user, self.final_item = users_emb_next, items_emb_next
        return users_emb_next, items_emb_next

class Graph_Comb_JGCF(nn.Module):
    def __init__(self, embed_dim):
        super(Graph_Comb_JGCF, self).__init__()
        #self.att_x = nn.Linear(embed_dim * 2, embed_dim * 2, bias=False)
        #self.att_y = nn.Linear(embed_dim * 2, embed_dim * 2, bias=False)
        self.comb = nn.Linear(embed_dim * 2, embed_dim * 2)
        nn.init.normal_(self.comb.weight, std=0.1)
        self.ratio_like = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x, y):
        #h1 = torch.tanh(self.att_x(x))
        #h2 = torch.tanh(self.att_y(y))
        #inputs = torch.hstack([x,y])
        #output = (x + y) / 2
        ratio = torch.sigmoid(self.ratio_like)
        output = ratio * x + (1.0 - ratio) * y
        #output = output / output.norm(2)
        return output         
            
class SocialSimGCL(SimGCL):

    def _init_weight(self):
        super(SocialSimGCL, self)._init_weight()
        self.socialGraph = self.dataset.getSocialGraph()
        self.Graph_Comb = Graph_Comb(self.latent_dim)
    

    def computer(self, perturbed=False):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        A = self.interactionGraph
        S = self.socialGraph

        embs = []
        for layer in range(self.n_layers):
            users_emb, items_emb = torch.split(all_emb, [self.num_users, self.num_items])
            users_emb_social = torch.sparse.mm(S, users_emb)
            all_emb_interaction = torch.sparse.mm(A, all_emb)
            if perturbed:
                users_emb_social = users_emb_social + self.random_noise(users_emb_social)
                all_emb_interaction = all_emb_interaction + self.random_noise(all_emb_interaction)
            users_emb_interaction, items_emb_next = torch.split(all_emb_interaction, [self.num_users, self.num_items])
            users_emb_next = self.Graph_Comb(users_emb_social, users_emb_interaction)
            all_emb = torch.cat([users_emb_next, items_emb_next])
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        final_embs = torch.mean(embs, dim=1)
        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        #if not perturbed:
        #    self.final_user, self.final_item = users, items
        return users, items


class SocialLGN(LightGCN):
    def _init_weight(self):
        super(SocialLGN, self)._init_weight()
        self.socialGraph = self.dataset.getSocialGraph()
        self.Graph_Comb = Graph_Comb(self.latent_dim)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        A = self.interactionGraph
        S = self.socialGraph
        embs = [all_emb]
        for layer in range(self.n_layers):
            # embedding from last layer
            users_emb, items_emb = torch.split(all_emb, [self.num_users, self.num_items])
            # social network propagation(user embedding)
            users_emb_social = torch.sparse.mm(S, users_emb)
            # user-item bi-network propagation(user and item embedding)
            all_emb_interaction = torch.sparse.mm(A, all_emb)
            # get users_emb_interaction
            users_emb_interaction, items_emb_next = torch.split(all_emb_interaction, [self.num_users, self.num_items])
            # graph fusion model
            users_emb_next = self.Graph_Comb(users_emb_social, users_emb_interaction)
            all_emb = torch.cat([users_emb_next, items_emb_next])
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        final_embs = torch.mean(embs, dim=1)
        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return users, items


class Graph_Comb(nn.Module):
    def __init__(self, embed_dim):
        super(Graph_Comb, self).__init__()
        self.att_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.att_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.comb = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x, y):
        h1 = torch.tanh(self.att_x(x))
        h2 = torch.tanh(self.att_y(y))
        output = self.comb(torch.cat((h1, h2), dim=1))
        output = output / output.norm(2)
        return output
