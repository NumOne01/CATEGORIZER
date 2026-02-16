import argparse
import os
import copy
import sys
import gc
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from functools import partial
from types import SimpleNamespace
from tqdm import tqdm

from vast import opensetAlgos
from vast.tools.pairwisedistances import cosine

import dataset
import utils
import losses
from net.resnet import *


def generate_dataset(dataset, index, index_target=None, target=None):
    """Generate a dataset by removing specified indices and optionally updating targets."""
    dataset_ = copy.deepcopy(dataset)

    if target is not None:
        for i, v in enumerate(index_target):
            dataset_.ys[v] = target[i]

    for i, v in enumerate(index):
        j = v - i
        dataset_.I.pop(j)
        dataset_.ys.pop(j)
        dataset_.im_paths.pop(j)
    return dataset_


def merge_dataset(dataset_o, dataset_n):
    """Merge two datasets by combining their indices, image paths, and labels."""
    dataset_ = copy.deepcopy(dataset_o)
    dataset_.I.extend(dataset_n.I)
    dataset_.im_paths.extend(dataset_n.im_paths)
    dataset_.ys.extend(dataset_n.ys)
    return dataset_


def common_processing(approach, params, criterion_pa, nb_classes, training_data):
    """Common processing function for openset algorithms."""
    global evm_args
    # Getting functions to execute
    param_fn = getattr(opensetAlgos, f'{approach}_Params')
    training_fn = getattr(opensetAlgos, f'{approach}_Training')
    testing_fn = getattr(opensetAlgos, f'{approach}_Inference')
    
    # Create argparse such that it gets all parameters provided by the Algo
    parser = argparse.ArgumentParser()
    parser, algo_params = param_fn(parser)
    evm_args = parser.parse_args(params.split())
            
    # Run the training function
    all_hyper_parameter_models = list(
        training_fn(
            pos_classes_to_process=training_data.keys(),
            features_all_classes=training_data,
            anchors={i: criterion_pa.proxies[i].detach().unsqueeze(0) for i in range(nb_classes)},
            args=evm_args,
            gpu=0,
            models=None
        )
    )
    # Assumes that there is only one hyper parameter combination and gets model for that combination
    models = dict(list(zip(*all_hyper_parameter_models))[1])
    
    return models


class EvtLoss(torch.nn.Module):
    """Extreme Value Theory Loss module."""
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
        
        P_one_hot = losses.binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)
        num_valid_proxies = len(with_pos_proxies)
        
        # Weibull loss (probability of sample inclusion)
        P_weibull_sum = torch.where(
            P_one_hot == 1, 1 - weibull_cdf, torch.zeros_like(weibull_cdf)
        ).sum(dim=0)
        N_weibull_sum = torch.where(
            N_one_hot == 1, weibull_cdf, torch.zeros_like(weibull_cdf)
        ).sum(dim=0)
        
        if num_valid_proxies == 0:
            num_valid_proxies = 1

        pos_term = torch.log(1 + P_weibull_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_weibull_sum).sum() / self.nb_classes
        loss = pos_term + neg_term
        
        return loss


def predict(X, extreme_vectors, scale, shape):
    """Predict using extreme vectors and Weibull parameters."""
    cos = cosine(X, extreme_vectors)
    weibull_cdf = torch.exp(
        -torch.pow(
            torch.abs(cos) / scale,
            shape,
        )
    )
    prediction = torch.argmax(weibull_cdf, dim=1)
    probs = torch.max(weibull_cdf, dim=1).values
    return prediction, probs


def set_cover(probabilities, cover_threshold):
    """Greedy set cover algorithm for selecting extreme vectors."""
    # threshold by cover threshold
    e = torch.eye(probabilities.shape[0]).type(torch.BoolTensor)
    thresholded = probabilities >= cover_threshold
    thresholded[e] = True
    del probabilities

    # greedily add points that cover most of the others
    covered = torch.zeros(thresholded.shape[0]).type(torch.bool)
    extreme_vectors = []
    covered_vectors = []

    while not torch.all(covered).item():
        sorted_indices = torch.topk(
            torch.sum(thresholded[:, ~covered], dim=1),
            len(extreme_vectors) + 1,
            sorted=False,
        ).indices
        for indx, sortedInd in enumerate(sorted_indices.tolist()):
            if sortedInd not in extreme_vectors:
                break
        else:
            print(thresholded.device, "ENTERING INFINITE LOOP ... EXITING")
            break
        covered_by_current_ev = torch.nonzero(thresholded[sortedInd, :], as_tuple=False)
        covered[covered_by_current_ev] = True
        extreme_vectors.append(sortedInd)
        covered_vectors.append(covered_by_current_ev.to("cpu"))
    
    del covered
    extreme_vectors_indexes = torch.tensor(extreme_vectors)
    return extreme_vectors_indexes, covered_vectors


def main():
    # Configuration
    args = {
        "LOG_DIR": "./logs",
        "dataset": "cub",
        "sz_embedding": 512,
        "sz_batch": 120,
        "nb_epochs": 60,
        "gpu_id": 0,
        "nb_workers": 4,
        "model": "resnet18",
        "loss": "Proxy_Anchor",
        "optimizer": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "lr_decay_step": 5,
        "lr_decay_gamma": 0.5,
        "alpha": 32,
        "mrg": 0.1,
        "warm": 5,
        "bn_freeze": True,
        "l2_norm": True,
        "remark": "",
        "use_split_modlue": True,
        "use_GM_clustering": True,
        "exp": "0",
    }
    args = SimpleNamespace(**args)

    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)

    # Setup paths
    pth_rst = "./result/" + args.dataset
    os.makedirs(pth_rst, exist_ok=True)
    pth_rst_exp = (
        pth_rst
        + "/"
        + args.model
        + "_sp_"
        + str(args.use_split_modlue)
        + "_gm_"
        + str(args.use_GM_clustering)
        + "_"
        + args.exp
    )
    os.makedirs(pth_rst_exp, exist_ok=True)

    # Dataset path
    pth_dataset = "./datasets"
    if args.dataset == "cub":
        pth_dataset += "/CUB200"
    elif args.dataset == "mit":
        pth_dataset += "/MIT67"
    elif args.dataset == "dog":
        pth_dataset += "/DOG120"
    elif args.dataset == "air":
        pth_dataset += "/AIR100"
    elif args.dataset == "cars":
        pth_dataset += "/CARS196"
    elif args.dataset == "cifar":
        pth_dataset += "/CIFAR"
    elif args.dataset == "cifar10":
        pth_dataset += "/CIFAR10"
    elif args.dataset == "svhn":
        pth_dataset += "/SVHN"
    elif args.dataset == "mnist":
        pth_dataset += "/MNIST"
    elif args.dataset == "mnist_fashion":
        pth_dataset += "/mnist_fashion"

    # Dataset Loader and Sampler
    dset_tr_0 = dataset.load(
        name=args.dataset,
        root=pth_dataset,
        mode="train_0",
        transform=dataset.utils.make_transform(is_train=True),
    )
    dlod_tr_0 = torch.utils.data.DataLoader(
        dset_tr_0, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers
    )
    nb_classes = dset_tr_0.nb_classes()

    # Backbone Model
    if args.model.find("resnet18") > -1:
        model = Resnet18(
            embedding_size=args.sz_embedding,
            pretrained=False,
            is_norm=args.l2_norm,
            bn_freeze=args.bn_freeze,
        )
    elif args.model.find("resnet50") > -1:
        model = Resnet50(
            embedding_size=args.sz_embedding,
            pretrained=False,
            is_norm=args.l2_norm,
            bn_freeze=args.bn_freeze,
            num_classes=None,
        )
    elif args.model.find("VIT") > -1:
        model = VIT(
            embedding_size=args.sz_embedding,
            pretrained=False,
            is_norm=args.l2_norm,
            bn_freeze=args.bn_freeze,
            num_classes=None,
        )
    else:
        print("Unknown model")
        sys.exit()

    model = model.cuda()

    # DML Losses
    criterion_pa = losses.Proxy_Anchor(
        nb_classes=nb_classes, 
        sz_embed=args.sz_embedding, 
        mrg=args.mrg, 
        alpha=args.alpha
    ).cuda()

    # Train Parameters
    param_groups = [
        {
            "params": (
                list(
                    set(model.parameters()).difference(
                        set(model.model.embedding.parameters())
                    )
                )
                if args.gpu_id != -1
                else list(
                    set(model.module.parameters()).difference(
                        set(model.module.model.embedding.parameters())
                    )
                )
            )
        },
        {
            "params": (
                model.model.embedding.parameters()
                if args.gpu_id != -1
                else model.module.model.embedding.parameters()
            ),
            "lr": float(args.lr) * 1,
        },
    ]
    param_groups.append({"params": criterion_pa.parameters(), "lr": float(args.lr) * 100})

    # Optimizer
    opt_pa = torch.optim.AdamW(
        param_groups, lr=float(args.lr), weight_decay=args.weight_decay
    )
    scheduler_pa = torch.optim.lr_scheduler.StepLR(
        opt_pa, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
    )

    print("Training parameters: {}".format(vars(args)))
    print("Training for {} epochs".format(args.nb_epochs))
    losses_list = []
    best_recall = [0]
    best_epoch = 0

    # Evaluation dataset
    dset_ev = dataset.load(
        name=args.dataset,
        root=pth_dataset,
        mode="eval_0",
        transform=dataset.utils.make_transform(is_train=False),
    )
    dlod_ev = torch.utils.data.DataLoader(
        dset_ev, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
    )

    # Training loop
    for epoch in range(0, args.nb_epochs):
        model.train()

        bn_freeze = args.bn_freeze
        if bn_freeze:
            modules = (
                model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
            )
            for m in modules:
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        losses_per_epoch = []

        # Warmup: Train only new params, helps stabilize learning.
        if args.warm > 0:
            if args.gpu_id != -1:
                unfreeze_model_param = list(model.model.embedding.parameters()) + list(
                    criterion_pa.parameters()
                )
            else:
                unfreeze_model_param = list(
                    model.module.model.embedding.parameters()
                ) + list(criterion_pa.parameters())

            if epoch == 0:
                for param in list(
                    set(model.parameters()).difference(set(unfreeze_model_param))
                ):
                    param.requires_grad = False
            if epoch == args.warm:
                for param in list(
                    set(model.parameters()).difference(set(unfreeze_model_param))
                ):
                    param.requires_grad = True

        total, correct = 0, 0
        pbar = tqdm(enumerate(dlod_tr_0))
        for batch_idx, (x, y, z) in pbar:
            feats = model(x.squeeze().cuda())
            loss_pa = criterion_pa(feats, y.squeeze().cuda())
            opt_pa.zero_grad()
            loss_pa.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            if args.loss == "Proxy_Anchor":
                torch.nn.utils.clip_grad_value_(criterion_pa.parameters(), 10)

            losses_per_epoch.append(loss_pa.data.cpu().numpy())
            opt_pa.step()

            pbar.set_description(
                "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}/{:.4f} Acc: {:.4f}".format(
                    epoch,
                    batch_idx + 1,
                    len(dlod_tr_0),
                    100.0 * batch_idx / len(dlod_tr_0),
                    loss_pa.item(),
                    0,
                    0,
                )
            )

        losses_list.append(np.mean(losses_per_epoch))
        scheduler_pa.step()

        if epoch >= 0:
            with torch.no_grad():
                print("Evaluating..")
                Recalls = utils.evaluate_cos(model, dlod_ev, epoch)

            # Best model save
            if best_recall[0] < Recalls[0]:
                best_recall = Recalls
                best_epoch = epoch
                torch.save(
                    {
                        "model_pa_state_dict": model.state_dict(),
                        "proxies_params": criterion_pa.proxies
                    },
                    "{}/{}_{}_best_step_0.pth".format(
                        pth_rst_exp, args.dataset, args.model
                    ),
                )
                with open(
                    "{}/{}_{}_best_results.txt".format(
                        pth_rst_exp, args.dataset, args.model
                    ),
                    "w",
                ) as f:
                    f.write(
                        "Best Epoch: {}\tBest Recall@{}: {:.4f}\n".format(
                            best_epoch, 1, best_recall[0] * 100
                        )
                    )

    # Load checkpoint
    print("==> Resuming from checkpoint..")
    pth_pth = (
        pth_rst_exp + "/" + "{}_{}_best_step_{}.pth".format(args.dataset, args.model, 0)
    )

    checkpoint = torch.load(pth_pth, weights_only=False)
    model.load_state_dict(checkpoint["model_pa_state_dict"])
    criterion_pa.proxies = checkpoint["proxies_params"]  # Fixed typo: proxies_param -> proxies_params

    model = model.cuda()
    model.eval()
    dlod_tr_evm = torch.utils.data.DataLoader(
        dset_tr_0, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
    )
    dset_ev = dataset.load(
        name=args.dataset,
        root=pth_dataset,
        mode="eval_0",
        transform=dataset.utils.make_transform(is_train=False),
    )
    dlod_ev = torch.utils.data.DataLoader(
        dset_ev, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
    )

    # Load EVM checkpoint
    pth_pth = (
        pth_rst_exp + "/" + "{}_{}_best_step_0_evm.pth".format(args.dataset, args.model)
    )
    checkpoint = torch.load(pth_pth, weights_only=False)
    model.load_state_dict(checkpoint["model_evt_state_dict"])
    extreme_vectors = checkpoint["extreme_vectors"]
    shape = checkpoint["shape"]
    scale = checkpoint["scale"]
    
    X_test, _ = utils.evaluate_cos_(model, dlod_ev)
    y_test = np.array(dlod_ev.dataset.ys)
    pred, prob = predict(X_test, extreme_vectors, scale, shape)

    acc_0 = (pred.cpu().numpy() == y_test).mean()
    print('Acc: ', acc_0)

    # Incremental learning setup
    args.nb_epochs = 30
    args.warm = 2
    args.steps = 1

    dlod_tr_prv = dlod_tr_0
    dset_tr_now_md = "train_1"
    dset_ev_now_md = "eval_1"
    nb_classes_prv = nb_classes
    nb_classes_evn = nb_classes

    dset_tr_now = dataset.load(
        name=args.dataset,
        root=pth_dataset,
        mode=dset_tr_now_md,
        transform=dataset.utils.make_transform(is_train=False),
    )
    dset_ev_now = dataset.load(
        name=args.dataset,
        root=pth_dataset,
        mode=dset_ev_now_md,
        transform=dataset.utils.make_transform(is_train=False),
    )
    dlod_tr_now = torch.utils.data.DataLoader(
        dset_tr_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
    )
    dlod_ev_now = torch.utils.data.DataLoader(
        dset_ev_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
    )

    # Calculate proxy mean and sigma for exemplar
    print("==> Calc. proxy mean and sigma for exemplar..")
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, dlod_tr_prv)
        feats = losses.l2_norm(feats)
        expler_s = feats.std(dim=0).cuda()

    with torch.no_grad():
        feats, labels = utils.evaluate_cos_(model, dlod_tr_now)

    print("Split using EVM..")
    preds_evm, probs_evm = predict(feats, extreme_vectors, scale, shape)
    probs_evm = probs_evm.detach().cpu().numpy()
    thres = 0.75
    utils.show_OnN(feats, labels, probs_evm, nb_classes_prv, pth_rst_exp, thres, False)

    preds_old = probs_evm[probs_evm >= thres]
    labels_old = labels[probs_evm >= thres]
    harder_thres_old = thres
    feats_old = torch.zeros(labels_old.shape)

    preds_new = probs_evm[probs_evm < thres]
    labels_new = labels[probs_evm < thres]
    harder_thres_new = thres
    feats_new = torch.zeros(labels_new.shape)

    probs_evm = torch.tensor(probs_evm).cuda()
    idx = torch.where(probs_evm >= harder_thres_old, 0, 1)
    idx_o = torch.nonzero(idx).squeeze()
    dset_tr_o = generate_dataset(dset_tr_now, idx_o)
    idx = torch.where(probs_evm < harder_thres_new, 0, 1)
    idx_n = torch.nonzero(idx).squeeze()
    dset_tr_n = generate_dataset(dset_tr_now, idx_n)
    dlod_tr_o = torch.utils.data.DataLoader(
        dset_tr_o, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
    )
    dlod_tr_n = torch.utils.data.DataLoader(
        dset_tr_n, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
    )

    print("==> Replace old labels..")
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, dlod_tr_o)
        preds_lb_o, _ = predict(feats, extreme_vectors, scale, shape)
        preds_lb_o = preds_lb_o.cpu().numpy()
        acc = np.mean(np.array(dlod_tr_o.dataset.ys) == preds_lb_o)
        print("Accuracy: {:.4f}".format(acc))

    print("==> Clustering splitted new and replace new labels..")
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, dlod_tr_n)
    
    clst_a = AffinityPropagation().fit(feats.cpu().numpy())
    p, c = np.unique(clst_a.labels_, return_counts=True)
    nb_classes_k = len(p)
    preds_lb_n = clst_a.labels_

    if args.use_GM_clustering:
        gm = GaussianMixture(
            n_components=nb_classes_k, max_iter=1000, tol=1e-4, init_params="kmeans"
        ).fit(feats.cpu().numpy())
        preds_lb_n = gm.predict(feats.cpu().numpy())

    dset_tr_now = dataset.load(
        name=args.dataset,
        root=pth_dataset,
        mode=dset_tr_now_md,
        transform=dataset.utils.make_transform(is_train=True),
    )
    dset_tr_o = generate_dataset(dset_tr_now, idx_o)
    dset_tr_n = generate_dataset(dset_tr_now, idx_n)
    dset_tr_o.ys = preds_lb_o.tolist()
    dset_tr_n.ys = (preds_lb_n + nb_classes_prv).tolist()
    dset_tr_now_m = merge_dataset(dset_tr_o, dset_tr_n)
    dlod_tr_now_m = torch.utils.data.DataLoader(
        dset_tr_now_m, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers
    )
    dlod_tr_now_evm = torch.utils.data.DataLoader(
        dset_tr_now_m, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers
    )

    gc.collect()

    print("==> Training splitted new..")
    nb_classes_now = nb_classes_prv + nb_classes_k
    criterion_pa_now = losses.Proxy_Anchor(
        nb_classes=nb_classes_now,
        sz_embed=args.sz_embedding,
        mrg=args.mrg,
        alpha=args.alpha,
    ).cuda()
    criterion_pa_now.proxies.data[:nb_classes_prv] = criterion_pa.proxies.data
    criterion_pa_now.proxies.data[nb_classes_prv:] = torch.from_numpy(
        clst_a.cluster_centers_
    ).cuda()

    bst_acc_a, bst_acc_oo, bst_acc_on, bst_acc_no, bst_acc_nn = 0.0, 0.0, 0.0, 0.0, 0.0
    bst_epoch_a, bst_epoch_o, bst_epoch_n = 0.0, 0.0, 0.0

    model_now = copy.deepcopy(model)
    model_now = model_now.cuda()

    param_groups = [
        {
            "params": (
                list(
                    set(model_now.parameters()).difference(
                        set(model_now.model.embedding.parameters())
                    )
                )
                if args.gpu_id != -1
                else list(
                    set(model_now.module.parameters()).difference(
                        set(model_now.module.model.embedding.parameters())
                    )
                )
            )
        },
        {
            "params": (
                model_now.model.embedding.parameters()
                if args.gpu_id != -1
                else model_now.module.model.embedding.parameters()
            ),
            "lr": float(args.lr) * 1,
        },
    ]
    param_groups.append(
        {"params": criterion_pa_now.parameters(), "lr": float(args.lr) * 100}
    )
    opt = torch.optim.AdamW(
        param_groups, lr=float(args.lr), weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
    )

    # Training loop for incremental learning
    for epoch in range(0, args.nb_epochs):
        model_now.train()

        bn_freeze = args.bn_freeze
        if bn_freeze:
            modules = (
                model_now.model.modules()
                if args.gpu_id != -1
                else model_now.module.model.modules()
            )
            for m in modules:
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        
        if args.warm > 0:
            if args.gpu_id != -1:
                unfreeze_model_param = list(model_now.model.embedding.parameters()) + list(
                    criterion_pa_now.parameters()
                )
            else:
                unfreeze_model_param = list(
                    model_now.module.model.embedding.parameters()
                ) + list(criterion_pa_now.parameters())

            if epoch == 0:
                for param in list(
                    set(model_now.parameters()).difference(set(unfreeze_model_param))
                ):
                    param.requires_grad = False
            if epoch == args.warm:
                for param in list(
                    set(model_now.parameters()).difference(set(unfreeze_model_param))
                ):
                    param.requires_grad = True

        pbar = tqdm(enumerate(dlod_tr_now_m))
        for batch_idx, (x, y, z) in pbar:
            feats = model_now(x.squeeze().cuda())

            # Exemplar generation using Weibull distribution
            y_n = torch.where(y > nb_classes_prv, 1, 0)
            y_o = y.size(0) - y_n.sum()
            if y_o > 0:
                y_sp = torch.randint(nb_classes_prv, (y_o,))
                feats_sp = torch.zeros(y_o, args.sz_embedding).cuda()
                for i, class_idx in enumerate(y_sp):
                    # Get proxy and Weibull parameters for this class
                    mean = criterion_pa.proxies[class_idx].detach()
                    shape_param = shape[class_idx].detach().item()
                    scale_param = scale[class_idx].detach().item()

                    # Sample distance from Weibull distribution
                    weibull_dist = torch.distributions.Weibull(
                        scale=torch.tensor(scale_param).cuda(),
                        concentration=torch.tensor(shape_param).cuda()
                    )
                    distance = weibull_dist.sample().cuda()
                    distance = torch.clamp(distance, 0, 1)

                    # Generate random unit vector
                    random_vec = torch.randn(1, args.sz_embedding).cuda()
                    random_vec = F.normalize(random_vec, p=2, dim=1)

                    # Normalize the proxy to unit length
                    mean_normalized = F.normalize(mean.unsqueeze(0), p=2, dim=1)

                    # Spherical interpolation: exemplar = mean * cos(angle) + random_vec * sin(angle)
                    target_sim = 1 - distance
                    # Make random_vec orthogonal to mean
                    random_vec = random_vec - torch.sum(random_vec * mean_normalized, dim=1, keepdim=True) * mean_normalized
                    random_vec = F.normalize(random_vec, p=2, dim=1)
                    # Compute angle and generate exemplar
                    angle = torch.acos(torch.clamp(target_sim, -1, 1))
                    exemplar = mean_normalized.squeeze() * torch.cos(angle) + random_vec.squeeze() * torch.sin(angle)
                    exemplar = F.normalize(exemplar.unsqueeze(0), p=2, dim=1)
                    feats_sp[i] = exemplar.squeeze()

                y = torch.cat((y, y_sp), dim=0)
                feats = torch.cat((feats, feats_sp), dim=0)
            
            loss_pa = criterion_pa_now(feats, y.squeeze().cuda())

            # Knowledge Distillation
            y_o_msk = torch.nonzero(y_n)
            if y_o_msk.size(0) > 1:
                y_o_msk = torch.nonzero(y_n).squeeze()
                x_o = torch.unsqueeze(x[y_o_msk[0]], dim=0)
                feats_n = torch.unsqueeze(feats[y_o_msk[0]], dim=0)
                for kd_idx in range(1, y_o_msk.size(0)):
                    x_o_ = torch.unsqueeze(x[y_o_msk[kd_idx]], dim=0)
                    x_o = torch.cat((x_o, x_o_), dim=0)
                    feats_n_ = torch.unsqueeze(feats[y_o_msk[kd_idx]], dim=0)
                    feats_n = torch.cat((feats_n, feats_n_), dim=0)
                with torch.no_grad():
                    feats_o = model(x_o.squeeze().cuda())
                feats_n = feats_n.cuda()
                loss_kd = torch.dist(
                    F.normalize(
                        feats_o.view(feats_o.size(0) * feats_o.size(1), 1), dim=0
                    ).detach(),
                    F.normalize(feats_n.view(feats_o.size(0) * feats_o.size(1), 1), dim=0),
                )
            else:
                loss_kd = torch.tensor(0.0).cuda()

            loss = loss_pa * 1.0 + loss_kd * 10.0

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_description(
                "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}/{:.6f}/{:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(dlod_tr_now_m),
                    100.0 * batch_idx / len(dlod_tr_now_m),
                    loss.item(),
                    loss_pa.item(),
                    loss_kd.item(),
                )
            )

        scheduler.step()

        # Generate exemplars for EVM training
        with torch.no_grad():
            X, _ = utils.evaluate_cos_(model_now, dlod_tr_now_evm)
        y = np.array(dlod_tr_now_evm.dataset.ys)

        for y_s in range(nb_classes_prv):
            mean = criterion_pa.proxies[y_s].detach()

            # Generate 50 samples using Weibull distribution
            shape_param = shape[y_s].detach().item()
            scale_param = scale[y_s].detach().item()

            weibull_dist = torch.distributions.Weibull(
                scale=torch.tensor(scale_param).cuda(),
                concentration=torch.tensor(shape_param).cuda()
            )
            distances = weibull_dist.sample((50,)).cuda()
            distances = torch.clamp(distances, 0, 1)

            random_vectors = torch.randn(50, args.sz_embedding).cuda()
            random_vectors = F.normalize(random_vectors, p=2, dim=1)

            mean_normalized = F.normalize(mean.unsqueeze(0), p=2, dim=1)

            exemplars = torch.zeros(50, args.sz_embedding).cuda()
            for i in range(50):
                target_sim = 1 - distances[i]
                random_vec = random_vectors[i:i+1]
                random_vec = random_vec - torch.sum(random_vec * mean_normalized, dim=1, keepdim=True) * mean_normalized
                random_vec = F.normalize(random_vec, p=2, dim=1)
                angle = torch.acos(torch.clamp(target_sim, -1, 1))
                exemplars[i] = mean_normalized.squeeze() * torch.cos(angle) + random_vec.squeeze() * torch.sin(angle)

            exemplars = F.normalize(exemplars, p=2, dim=1)

            X = torch.cat((X, exemplars), dim=0)
            y = np.append(y, np.full(50, y_s))

        training_data = {i: X[y == i] for i in range(nb_classes_now)}

        models_now = common_processing(
            "EVM",
            params="--distance_metric cosine --tailsize 1000 --distance_multiplier 1 --cover_threshold 1",
            criterion_pa=criterion_pa_now,
            nb_classes=nb_classes_now,
            training_data=training_data,
        )

        extreme_vectors_now = torch.cat(
            [value.get("extreme_vectors") for value in models_now.values()], dim=0
        ).cuda()
        weibull_now = torch.cat(
            [value.get("weibulls").wbFits for value in models_now.values()], dim=0
        ).cuda()
        scale_now = weibull_now[:, 1]
        shape_now = weibull_now[:, 0]

        # Apply set cover to reduce extreme vectors
        distances = cosine(extreme_vectors_now[nb_classes_prv:], extreme_vectors_now[nb_classes_prv:])
        probs = torch.exp(
            -torch.pow(
                torch.abs(distances) / scale_now[nb_classes_prv:],
                shape_now[nb_classes_prv:],
            )
        )

        indices, covered_vectors = set_cover(probs, 0.9999)

        extreme_vectors_now = torch.cat((
            extreme_vectors_now[:nb_classes_prv], 
            extreme_vectors_now[nb_classes_prv:][indices]
        ))
        scale_now = torch.cat((
            scale_now[:nb_classes_prv], 
            scale_now[nb_classes_prv:][indices]
        ))
        shape_now = torch.cat((
            shape_now[:nb_classes_prv], 
            shape_now[nb_classes_prv:][indices]
        ))

        print(len(extreme_vectors_now))

        # Evaluation
        print("==> Evaluation..")
        model.eval()
        model_now.eval()
        with torch.no_grad():
            feats, _ = utils.evaluate_cos_(model_now, dlod_ev_now)
            preds_lb, _ = predict(feats, extreme_vectors_now, scale_now, shape_now)
            preds_lb = preds_lb.cpu().numpy()

            y = np.array(dlod_ev_now.dataset.ys)

            proj_all_new = utils.cluster_pred_2_gt(preds_lb.astype(int), y.astype(int))
            pacc_fun_all_new = partial(utils.pred_2_gt_proj_acc, proj_all_new)
            acc_a = pacc_fun_all_new(y.astype(int), preds_lb.astype(int))

            selected_mask = y < nb_classes
            acc_o = pacc_fun_all_new(
                y[selected_mask].astype(int), preds_lb[selected_mask].astype(int)
            )
            selected_mask = y >= nb_classes_evn
            acc_n = pacc_fun_all_new(
                y[selected_mask].astype(int), preds_lb[selected_mask].astype(int)
            )

        if acc_a > bst_acc_a:
            bst_acc_a = acc_a
            bst_epoch_a = epoch

        if acc_o > bst_acc_oo:
            bst_acc_on = acc_n
            bst_acc_oo = acc_o
            bst_epoch_o = epoch
        if acc_n > bst_acc_nn:
            bst_acc_nn = acc_n
            bst_acc_no = acc_o
            bst_epoch_n = epoch

        print(
            "Valid Epoch: {} Acc: {:.4f}/{:.4f}/{:.4f} Best result: {}/{}/{} {:.4f}/{:.4f}/{:.4f}".format(
                epoch,
                acc_a,
                acc_o,
                acc_n,
                bst_epoch_a,
                bst_epoch_o,
                bst_epoch_n,
                bst_acc_a,
                bst_acc_oo,
                bst_acc_nn,
            )
        )
        pth_rst_exp_log = pth_rst_exp + "/" + "result_evm.txt"
        with open(pth_rst_exp_log, "a+") as fval:
            fval.write(
                "Valid Epoch: {} Acc: {:.4f}/{:.4f}/{:.4f}/{:.4f} Best result: {}/{}/{} {:.4f}/{:.4f}/{:.4f}\n".format(
                    epoch,
                    0,
                    acc_a,
                    acc_o,
                    acc_n,
                    bst_epoch_a,
                    bst_epoch_o,
                    bst_epoch_n,
                    bst_acc_a,
                    bst_acc_oo,
                    bst_acc_nn,
                )
            )

        step = 1


if __name__ == "__main__":
    main()

