import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses
from vast.tools.pairwisedistances import cosine

import sys


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing

    T = sklearn.preprocessing.label_binarize(T, classes=range(0, nb_classes))
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

        self.weibull_shape = torch.nn.Parameter(
            torch.randn(nb_classes).cuda()
        )  # Shape (k)
        self.weibull_scale = torch.nn.Parameter(
            torch.randn(nb_classes).cuda()
        )  # Scale (lambda)

    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity

        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1
        )  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(
            dim=0
        )
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(
            dim=0
        )
        
        # Weibull loss (probability of sample inclusion)
        # weibull_cdf = torch.exp(
        #     -torch.pow(
        #         torch.abs(cos / self.weibull_scale),
        #         self.weibull_shape,
        #     )
        # )
        # P_weibull_sum = torch.where(P_one_hot == 1, 1 - weibull_cdf, torch.zeros_like(weibull_cdf)).sum(
        #     dim=0
        # )
        # N_weibull_sum = torch.where(N_one_hot == 1, weibull_cdf, torch.zeros_like(weibull_cdf)).sum(
        #     dim=0
        # )

        if num_valid_proxies == 0:
            num_valid_proxies = 1

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss


# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(
            num_classes=self.nb_classes,
            embedding_size=self.sz_embed,
            softmax_scale=self.scale,
        ).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class MultiSimilarityLoss(torch.nn.Module):
    def __init__(
        self,
    ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50

        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(
            self.scale_pos, self.scale_neg, self.thresh
        )

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets="semihard")
        self.loss_func = losses.TripletMarginLoss(margin=self.margin)

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(
            l2_reg_weight=self.l2_reg, normalize_embeddings=False
        )

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class MultiProxyLoss(torch.nn.Module):
    def __init__(
        self,
        nb_classes,
        sz_embed,
        mrg=0.1,
        alpha=32,
        num_proxies_per_class=1,
        temperature=1,
        gamma=0.5,
    ):
        torch.nn.Module.__init__(self)
        self.proxies = nn.Parameter(
            torch.randn(nb_classes, num_proxies_per_class, sz_embed)
        )
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.num_proxies_per_class = num_proxies_per_class
        self.temperature = temperature
        self.gamma = gamma

    def forward(self, X, T):
        P = self.proxies.view(
            self.nb_classes * self.num_proxies_per_class, self.sz_embed
        )
        self.gamma = 1 / 120

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity

        confidence = cos.view(-1, self.nb_classes, self.num_proxies_per_class)

        confidence = torch.exp(confidence / self.temperature)

        confidence = confidence / confidence.sum(dim=2, keepdim=True)

        similarity = cos.view(-1, self.nb_classes, self.num_proxies_per_class)

        pssitive_simmilarity = torch.where(
            confidence > self.gamma, similarity, torch.zeros_like(confidence)
        )

        pssitive_simmilarity = torch.exp(
            -self.alpha * (pssitive_simmilarity - self.mrg)
        )

        pssitive_simmilarity_sum = pssitive_simmilarity.mean(dim=2)

        pos_exp = pssitive_simmilarity_sum

        non_positive_simmilarity = torch.where(
            confidence <= self.gamma, similarity, torch.zeros_like(confidence)
        )

        maximum_non_positive_simmilarity = non_positive_simmilarity.max(
            dim=2, keepdim=True
        )[0]

        negative_simmilarity = torch.where(
            similarity > maximum_non_positive_simmilarity,
            similarity,
            torch.zeros_like(confidence),
        )

        negative_simmilarity = torch.exp(self.alpha * (negative_simmilarity - self.mrg))

        negative_simmilarity_sum = negative_simmilarity.mean(dim=2)

        neg_exp = negative_simmilarity_sum

        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1
        )  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(
            dim=0
        )
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(
            dim=0
        )

        # Compute cosine similarity between proxies
        proxy_cos = F.linear(l2_norm(P), l2_norm(P))  # proxy-to-proxy similarity

        # Get the same-class and different-class masks
        same_class_mask = (
            torch.eye(self.nb_classes * self.num_proxies_per_class).bool().to(P.device)
        )
        diff_class_mask = ~same_class_mask

        # Apply the temperature scaling
        # proxy_conf = torch.exp(proxy_cos / self.temperature)

        # proxy_conf = proxy_conf.view(-1, self.nb_classes, self.num_proxies_per_class)
        # proxy_conf = proxy_conf / proxy_conf.sum(dim=2, keepdim=True)

        proxy_similarities = proxy_cos.view(
            -1, self.nb_classes, self.num_proxies_per_class
        )

        non_negative_simmilarity = torch.where(
            similarity <= maximum_non_positive_simmilarity,
            similarity,
            torch.zeros_like(confidence),
        )

        maximum_non_negative_simmilarity = non_negative_simmilarity.max(
            dim=0, keepdim=True
        )[0]

        proxy_similarities = torch.where(
            proxy_similarities > maximum_non_negative_simmilarity,
            proxy_similarities,
            torch.zeros_like(proxy_similarities),
        )

        proxy_similarities = proxy_similarities.view(
            -1, self.nb_classes * self.num_proxies_per_class
        )

        inter_class_sim = proxy_similarities * diff_class_mask.float()

        inter_class_sim = torch.exp(self.alpha * (inter_class_sim - self.mrg))

        # Encourage inter-class separation
        inter_class_term = torch.log(1 + inter_class_sim.sum()).sum() / self.nb_classes

        if num_valid_proxies == 0:
            num_valid_proxies = 1

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes

        loss = pos_term + neg_term + inter_class_term

        # Combine the terms into the final loss

        return loss

class EvtLoss(torch.nn.Module):
    def __init__(self, nb_classes, extreme_vectors, shape, scale):
        torch.nn.Module.__init__(self)
        self.nb_classes = nb_classes

        self.extreme_vectors = nn.Parameter(extreme_vectors)
        self.shape = nn.Parameter(shape)
        self.scale = nn.Parameter(scale)
        

    def forward(self, X, T):
        cos = cosine(X, self.extreme_vectors)
        weibull_cdf = torch.exp(
            -torch.pow(
                torch.abs(cos) / self.scale,
                self.shape,
            )
        )
        
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1
        )  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies
        # Weibull loss (probability of sample inclusion)
        P_weibull_sum = torch.where(P_one_hot == 1, 1 - weibull_cdf, torch.zeros_like(weibull_cdf)).sum(
            dim=0
        )
        N_weibull_sum = torch.where(N_one_hot == 1, weibull_cdf, torch.zeros_like(weibull_cdf)).sum(
            dim=0
        )
        if num_valid_proxies == 0:
            num_valid_proxies = 1

        pos_term = torch.log(1 + P_weibull_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_weibull_sum).sum() / self.nb_classes
        loss = pos_term + neg_term
        
        return loss