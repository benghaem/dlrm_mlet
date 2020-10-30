# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import bisect
import builtins
import shutil
import time
import sys

# data generation
import dlrm_data_pytorch as dp
import dlrm_data_avazu_pytorch as dp_ava

# numpy
import numpy as np

# pickle
import pickle

#nvidia apex
# from apex import amp

# onnx
import onnx

# pytorch
import torch
import torch.nn as nn
from numpy import random as ra
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter

from sklearn.metrics import roc_auc_score

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

exc = getattr(builtins, "IOError", "FileNotFoundError")


### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        for i in range(0, ln.size):
            n = ln[i]

            # construct embedding operator
            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            # initialize embeddings
            # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            # approach 1
            EE.weight.data = torch.tensor(W, requires_grad=True)
            # approach 2
            # EE.weight.data.copy_(torch.tensor(W))
            # approach 3
            # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
            emb_l.append(EE)

        return emb_l

    def create_linp(self, in_feat, out_feat, table_count):
        emb_linp = nn.ModuleList()

        print("Building linp {} -> {}".format(in_feat, out_feat))
        for i in range(0, table_count):

            LIN = nn.Linear(in_feat, out_feat, bias=False)

            mode_args = self.linp_init.split("-")
            mode = mode_args[0]

            if (mode == "normal"):
                stdev = 0.25
                if len(mode_args) == 2:
                    stdev = float(mode_args[1])
                nn.init.normal_(LIN.weight, mean=0.0, std=stdev)
            elif (mode == "xavier"):
                nn.init.xavier_normal_(LIN.weight)
            elif (mode =="rp"):
                LIN.weight.data = torch.transpose(torch.tensor(self.rp_mats[i],
                        requires_grad=True), 0, 1)

            emb_linp.append(LIN)

        return emb_linp

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        enable_rp = False,
        enable_rp_up = False,
        rp_mats = None,
        rpup_mats = None,
        enable_linp = False,
        enable_linp_up = False,
        proj_down_dim = -1,
        proj_up_dim = -1,
        linp_init = None,
        concat_og_feat = False
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            # create operators
            self.emb_l = self.create_emb(m_spa, ln_emb)
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

            self.enable_rp = enable_rp
            self.enable_rp_up = enable_rp_up
            self.rp_mats = rp_mats
            self.rpup_mats = rpup_mats

            self.enable_linp = enable_linp
            self.enable_linp_up = enable_linp_up
            if (self.enable_linp_up):
                self.proj_down_dim = proj_down_dim
            else:
                self.proj_down_dim = ln_bot[ln_bot.size -1]

            if (proj_up_dim != -1):
                self.proj_up_dim = proj_up_dim
            else:
                self.proj_up_dim = m_spa

            self.linp_init = linp_init
            if (self.enable_linp):
                self.emb_linp = self.create_linp(m_spa,
                                                 self.proj_down_dim,
                                                 ln_emb.size)
            if (self.enable_linp_up):
                self.emb_linp_up = self.create_linp(self.proj_down_dim,
                                                    self.proj_up_dim,
                                                    ln_emb.size)
            self.concat_og_feat = concat_og_feat

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_linp(self, ly, layers):
        ly_pro = []
        act_fn = nn.Identity()

        for var_id, ly_o in enumerate(ly):
            ly_pro.append(act_fn(layers[var_id](ly_o)))

        return ly_pro

    # def apply_linp_up(self, ly, layers):
    #     ly_pro = []
    #     act_fn = nn.Identity()

    #     for var_id, ly_o in enumerate(ly):
    #         ly_pro.append(act_fn(layers[var_id](ly_o)))

    def apply_emb(self, lS_o, lS_i, emb_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            E = emb_l[k]
            V = E(sparse_index_group_batch, sparse_offset_group_batch)

            # if we are using random projection mode, apply it here
            if self.enable_rp:
                RP = self.rp_mats[k]
                V_RP = torch.matmul(V,RP)
                if self.enable_rp_up:
                    RPUP = self.rpup_mats[k]
                    V_RP_RPUP = torch.matmul(V_RP, RPUP)
                    ly.append(V_RP_RPUP)
                else:
                    ly.append(V_RP)
            else:
                ly.append(V)

        # print(ly)
        return ly

    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, lS_i):
        if self.ndevices <= 1:
            return self.sequential_forward(dense_x, lS_o, lS_i)
        else:
            return self.parallel_forward(dense_x, lS_o, lS_i)

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l)

        if (self.enable_linp):
            ly_pro = self.apply_linp(ly, self.emb_linp)
            if (self.enable_linp_up):
                ly_pro = self.apply_linp(ly_pro, self.emb_linp_up)
        else:
            ly_pro = ly
            if (self.enable_linp_up):
                ly_pro = self.apply_linp(ly_pro, self.emb_linp_up)

        # interact features (dense and sparse)
        z = self.interact_features(x, ly_pro)
        # print(z.detach().cpu().numpy())

        # concatenate features onto output
        if (self.concat_og_feat):
            #print(z.shape, dense_x.shape, [emb.shape for emb in ly_pro])
            z = torch.cat((z, dense_x, *ly_pro), 1)

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

    def parallel_forward(self, dense_x, lS_o, lS_i):
        ### prepare model (overwrite) ###
        # WARNING: # of devices must be >= batch size in parallel_forward call
        batch_size = dense_x.size()[0]
        ndevices = min(self.ndevices, batch_size, len(self.emb_l))
        device_ids = range(ndevices)
        # WARNING: must redistribute the model if mini-batch size changes(this is common
        # for last mini-batch, when # of elements in the dataset/batch size is not even
        if self.parallel_model_batch_size != batch_size:
            self.parallel_model_is_not_prepared = True

        if self.sync_dense_params or self.parallel_model_is_not_prepared:
            # replicate mlp (data parallelism)
            self.bot_l_replicas = replicate(self.bot_l, device_ids)
            self.top_l_replicas = replicate(self.top_l, device_ids)
            # distribute embeddings (model parallelism)
            t_list = []
            for k, emb in enumerate(self.emb_l):
                d = torch.device("cuda:" + str(k % ndevices))
                emb.to(d)
                t_list.append(emb.to(d))
            self.emb_l = nn.ModuleList(t_list)
            self.parallel_model_batch_size = batch_size
            self.parallel_model_is_not_prepared = False

        ### prepare input (overwrite) ###
        # scatter dense features (data parallelism)
        # print(dense_x.device)
        dense_x = scatter(dense_x, device_ids, dim=0)
        # distribute sparse features (model parallelism)
        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit("ERROR: corrupted model input detected in parallel_forward call")

        t_list = []
        i_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            t_list.append(lS_o[k].to(d))
            i_list.append(lS_i[k].to(d))
        lS_o = t_list
        lS_i = i_list

        ### compute results in parallel ###
        # bottom mlp
        # WARNING: Note that the self.bot_l is a list of bottom mlp modules
        # that have been replicated across devices, while dense_x is a tuple of dense
        # inputs that has been scattered across devices on the first (batch) dimension.
        # The output is a list of tensors scattered across devices according to the
        # distribution of dense_x.
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
        # debug prints
        # print(x)

        # embeddings
        ly = self.apply_emb(lS_o, lS_i, self.emb_l)
        # debug prints
        # print(ly)

        # butterfly shuffle (implemented inefficiently for now)
        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on each device. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each device.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        if len(self.emb_l) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        t_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)
        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))
        # debug prints
        # print(ly)

        # interactions
        z = []
        for k in range(ndevices):
            zk = self.interact_features(x[k], ly[k])
            z.append(zk)
        # debug prints
        # print(z)

        # top mlp
        # WARNING: Note that the self.top_l is a list of top mlp modules that
        # have been replicated across devices, while z is a list of interaction results
        # that by construction are scattered across devices on the first (batch) dim.
        # The output is a list of tensors scattered across devices according to the
        # distribution of z.
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)

        ### gather the distributed results ###
        p0 = gather(p, self.output_d, dim=0)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0

        return z0


if __name__ == "__main__":
    ### import packages ###
    import sys
    import os
    import io
    import collections
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument("--arch-embedding-size", type=str, default="4-3-2")
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=str, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=str, default="4-2-1")
    parser.add_argument("--arch-interaction-op", type=str, default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or avazu
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)

    # dump_emb_init
    parser.add_argument("--dump-emb-init", action="store_true", default=False)

    # random projection
    parser.add_argument("--enable-rp", action="store_true", default=False)
    parser.add_argument("--rp-file", type=str, default="")
    parser.add_argument("--enable-rp-up", action="store_true", default=False)
    parser.add_argument("--rpup-file", type=str, default="")

    # concat_og_output
    parser.add_argument("--concat-og-features", action="store_true",
            default=False)

    # linear projection
    parser.add_argument("--enable-linp", action="store_true", default=False)
    parser.add_argument("--enable-linp-up", action="store_true", default=False)
    parser.add_argument("--proj-down-dim", type=int, default=4)
    parser.add_argument("--proj-up-dim", type=int, default=-1)
    # normal, xaiver, rp
    parser.add_argument("--linp-init", type=str, default="normal")
    # none, sigmoid
    parser.add_argument("--linp-act", type=str, default="none")

    # half_precision
    parser.add_argument("--fp16", action="store_true", default=False)

    # avazu database
    parser.add_argument("--avazu-db-path", type=str, default="")
    parser.add_argument("--avazu-epoch-shuffle", action="store_true", default=False)

    # apex mode
    parser.add_argument("--enable_amp",action="store_true", default=False)
    parser.add_argument("--apex-mode", type=str, default="O0")

    # auc metric
    parser.add_argument("--auc-only",action="store_true", default=False)
    parser.add_argument("--enable-auc",action="store_true", default=True)

    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)

    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    args = parser.parse_args()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    use_gpu = args.use_gpu and torch.cuda.is_available()
    use_fp16 = args.fp16 and use_gpu

    print("---INIT DLRM---")

    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda", 0)
        ngpus = torch.cuda.device_count()  # 1
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare RP matrcies ###
    rp_mats = None
    rpup_mats = None
    if args.enable_rp:
        if use_gpu:
            rp_mats = torch.tensor(pickle.load(open(args.rp_file,"rb")),
                                   device=device)
        else:
            rp_mats = torch.tensor(pickle.load(open(args.rp_file,"rb")))

    if args.enable_rp_up:
        if use_gpu:
            rpup_mats = torch.tensor(pickle.load(open(args.rpup_file, "rb")),
                                    device=device)
        else:
            rpup_mats = torch.tensor(pickle.load(open(args.rpup_file, "rb")))

    if (use_fp16 or args.apex_mode != "O0") and args.enable_rp:
        rp_mats = rp_mats.half()

    if (use_fp16 or args.apex_mode != "O0") and args.enable_rp_up:
        rpup_mats = rpup_mats.half()

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data
    if args.data_generation == "dataset":
        # input and target from dataset
        def collate_wrapper(list_of_tuples):
            # where each tuple is (X_int, X_cat, y)
            transposed_data = list(zip(*list_of_tuples))
            X_int = torch.stack(transposed_data[0], 0)
            X_cat = torch.stack(transposed_data[1], 0)
            T     = torch.stack(transposed_data[2], 0).view(-1,1)

            sz0 = X_cat.shape[0]
            sz1 = X_cat.shape[1]
            if use_gpu:
                lS_i = [X_cat[:, i].pin_memory() for i in range(sz1)]
                lS_o = [torch.tensor(range(sz0)).pin_memory() for _ in range(sz1)]
                return X_int.pin_memory(), lS_o, lS_i, T.pin_memory()
            else:
                lS_i = [X_cat[:, i] for i in range(sz1)]
                lS_o = [torch.tensor(range(sz0)) for _ in range(sz1)]
                return X_int, lS_o, lS_i, T

        train_data = None
        test_data = None

        if (args.data_set == "kaggle" or args.data_set == "terabyte"):
            train_data = dp.CriteoDataset(
                args.data_set,
                args.data_randomize,
                "train",
                args.raw_data_file,
                args.processed_data_file,
            )

            test_data = dp.CriteoDataset(
                args.data_set,
                args.data_randomize,
                "test",
                args.raw_data_file,
                args.processed_data_file,
            )
        if (args.data_set == "avazu"):
            train_data = dp_ava.AvazuDataset(
                args.avazu_db_path,
                split = 'train',
                dup_to_mem = False,
                #chunk_size=3000000
                chunk_size=1000000
            )
            test_data = dp_ava.AvazuDataset(
                args.avazu_db_path,
                split = 'val',
                dup_to_mem = False,
                chunk_size=5000000
                #chunk_size=20000
            )

        #report model params
        ln_emb = train_data.counts
        m_den = train_data.m_den
        ln_bot[0] = m_den



        #setup loaders
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.mini_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_wrapper,
            pin_memory=False,
            drop_last=False,
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.mini_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_wrapper,
            pin_memory=False,
            drop_last=False,
        )




        #setup batch sizes
        nbatches = args.num_batches if args.num_batches > 0 else len(train_loader)
        nbatches_test = len(test_loader)
    else:
        # input and target at random
        def collate_wrapper(list_of_tuples):
            # where each tuple is (X, lS_o, lS_i, T)
            if use_gpu:
                (X, lS_o, lS_i, T) = list_of_tuples[0]
                return (X.pin_memory(),
                        [S_o.pin_memory() for S_o in lS_o],
                        [S_i.pin_memory() for S_i in lS_i],
                        T.pin_memory())
            else:
                return list_of_tuples[0]

        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data = dp.RandomDataset(
            m_den,
            ln_emb,
            args.data_size,
            args.num_batches,
            args.mini_batch_size,
            args.num_indices_per_lookup,
            args.num_indices_per_lookup_fixed,
            1, # num_targets
            args.round_targets,
            args.data_generation,
            args.data_trace_file,
            args.data_trace_enable_padding,
            reset_seed_on_access=True,
            rand_seed=args.numpy_rand_seed
        ) #WARNING: generates a batch of lookups at once
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_wrapper,
            pin_memory=False,
            drop_last=False,
        )
        nbatches = args.num_batches if args.num_batches > 0 else len(train_loader)

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    num_fea = ln_emb.size + 1  # num sparse + num dense features
    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )

    first_top_dim = num_int
    if args.concat_og_features:
        first_top_dim = (num_int + (ln_emb.size) * m_den_out + ln_bot[0])

    arch_mlp_top_adjusted = str(first_top_dim) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )

    #TODO: Fix this check to match sizes
    if m_spa != m_den_out and \
       args.enable_rp == False and \
       args.enable_linp == False:
        sys.exit(
            "ERROR: arch-sparse-feature-size "
            + str(m_spa)
            + " does not match last dim of bottom mlp "
            + str(m_den_out)
        )

    if num_int != ln_top[0] and not args.concat_og_features:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # test prints (model arch)
    if args.debug_mode:
        print("model arch:")
        print(
            "mlp top arch "
            + str(ln_top.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_top)
        print("# of interactions")
        print(num_int)
        print(
            "mlp bot arch "
            + str(ln_bot.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_bot)
        print("# of features (sparse and dense)")
        print(num_fea)
        print("dense feature size")
        print(m_den)
        print("sparse feature size")
        print(m_spa)
        print(
            "# of embeddings (= # of sparse features) "
            + str(ln_emb.size)
            + ", with dimensions "
            + str(m_spa)
            + "x:"
        )
        print(ln_emb)

        print("data (inputs and targets):")
        for j, (X, lS_o, lS_i, T) in enumerate(train_loader):
            # early exit if nbatches was set by the user and has been exceeded
            if j >= nbatches:
                break

            print("mini-batch: %d" % j)
            print(X.detach().cpu().numpy())
            # transform offsets to lengths when printing
            print(
                [
                    np.diff(
                        S_o.detach().cpu().tolist() + list(lS_i[i].shape)
                    ).tolist()
                    for i, S_o in enumerate(lS_o)
                ]
            )
            print([S_i.detach().cpu().tolist() for S_i in lS_i])
            print(T.detach().cpu().numpy())

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        enable_rp=args.enable_rp,
        enable_rp_up=args.enable_rp_up,
        rp_mats=rp_mats,
        rpup_mats=rpup_mats,
        enable_linp=args.enable_linp,
        enable_linp_up=args.enable_linp_up,
        proj_down_dim=args.proj_down_dim,
        proj_up_dim=args.proj_up_dim,
        linp_init=args.linp_init,
        concat_og_feat=args.concat_og_features,
    )
    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())
        # print(dlrm)

    if use_gpu:
        if ngpus > 1:
            # Custom Model-Data Parallel
            # the mlps are replicated and use data parallelism, while
            # the embeddings are distributed and use model parallelism
            dlrm.ndevices = min(ngpus, args.mini_batch_size, num_fea - 1)
        dlrm = dlrm.to(device)  # .cuda()

    if use_fp16:
        dlrm = dlrm.half()

    # specify the loss function
    if args.loss_function == "mse":
        loss_fn = torch.nn.MSELoss(reduction="mean")
    elif args.loss_function == "bce":
        loss_fn = torch.nn.BCELoss(reduction="mean")
    else:
        sys.exit("ERROR: --loss-function=" + args.loss_function + " is not supported")

    if args.dump_emb_init:
        with open("/tmp/dlrm_emb_table_dump.emb", "wb+") as f:
            pickle.dump(dlrm.emb_l, f)
        if args.enable_linp:
            with open("/tmp/dlrm_emb_mat_dump.emb", "wb+") as f:
                pickle.dump(dlrm.emb_linp, f)
        sys.exit()

    if not args.inference_only:
        # specify the optimizer algorithm
        optimizer = torch.optim.SGD(dlrm.parameters(), lr=args.learning_rate)

        #use apex
        if args.enable_amp:
            dlrm, optimizer = amp.initialize(dlrm, optimizer,
                    opt_level=args.apex_mode)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=5,
                                                    gamma=0.1)

    ### main loop ###
    def time_wrap(use_gpu):
        if use_gpu:
            torch.cuda.synchronize()
        return time.time()

    def dlrm_wrap(X, lS_o, lS_i, use_gpu, use_fp16, device):
        if use_gpu:  # .cuda()
            if use_fp16:
                return dlrm(
                    X.to(device).half(),
                    [S_o.to(device) for S_o in lS_o],
                    [S_i.to(device) for S_i in lS_i],
                )
            else:
                return dlrm(
                    X.to(device),
                    [S_o.to(device) for S_o in lS_o],
                    [S_i.to(device) for S_i in lS_i],
                )
        else:
            return dlrm(X, lS_o, lS_i)

    def loss_fn_wrap(Z, T, use_gpu, use_fp16, device):
        if use_gpu:
            if use_fp16:
                Td = T.to(device).half()
                assert (Z >= 0.).all() and (Z <= 1.).all()
                assert (T >= 0.).all() and (T <= 1.).all()
                return loss_fn(Z, Td)
            else:
                return loss_fn(Z, T.to(device))
        else:
            return loss_fn(Z, T)

    # training or inference
    best_gA_test = 0
    best_gL_test = 1
    best_auc = 0.0
    total_time = 0
    total_loss = 0
    total_accu = 0
    total_iter = 0
    k = 0

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved mode {}".format(args.load_model))
        ld_model = torch.load(args.load_model)
        dlrm.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_gA = ld_model["train_acc"]
        ld_gL = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        ld_total_accu = ld_model["total_accu"]
        ld_gA_test = ld_model["test_acc"]
        ld_gL_test = ld_model["test_loss"]
        ld_auc = ld_model["auc"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_gA_test = ld_gA_test
            best_gL_test = ld_gL_test
            best_auc = ld_auc
            total_loss = ld_total_loss
            total_accu = ld_total_accu
            k = ld_k  # epochs
            j = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0
        print(
            "Saved model Training state: epoch = {:d}/{:d}, batch = {:d}/{:d}, train loss = {:.6f}, train accuracy = {:3.3f} %".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_gL, ld_gA * 100
            )
        )
        print(
            "Saved model Testing state: nbatches = {:d}, test loss = {:.6f}, test accuracy = {:3.3f} %".format(
                ld_nbatches_test, ld_gL_test, ld_gA_test * 100
            )
        )
        if (args.auc_only):
            test_accu = 0
            test_loss = 0
            size = args.mini_batch_size * len(test_loader)
            print("size of test set: " + str(size))
            y_true = np.zeros(size)
            y_score = np.zeros(size)
            for jt, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(test_loader):
                # forward pass
                Z_test = dlrm_wrap(
                        X_test, lS_o_test, lS_i_test, use_gpu, use_fp16, device
                        )
                Z_np = Z_test.detach().cpu().numpy().reshape(1, len(Z_test))
                T_np = T_test.detach().cpu().numpy().reshape(1, len(T_test))
                b = jt * len(Z_test)
                e = b + len(Z_test)
                y_true[b:e] = T_np[:]
                y_score[b:e] = Z_np[:]

            auc = roc_auc_score(y_true, y_score)
            print("AUC = {:.12f}".format(auc))
            sys.exit()

    print("time/loss/accuracy (if enabled):")
    with torch.autograd.profiler.profile(args.enable_profiling, use_gpu) as prof:
        while k < args.nepochs:
            for j, (X, lS_o, lS_i, T) in enumerate(train_loader):
                # early exit if nbatches was set by the user and has been exceeded
                if j >= nbatches:
                    break
                '''
                # debug prints
                print("input and targets")
                print(X.detach().cpu().numpy())
                print([np.diff(S_o.detach().cpu().tolist() + list(lS_i[i].shape)).tolist() for i, S_o in enumerate(lS_o)])
                print([S_i.detach().cpu().numpy().tolist() for S_i in lS_i])
                print(T.detach().cpu().numpy())
                '''
                t1 = time_wrap(use_gpu)

                # forward pass
                Z = dlrm_wrap(X, lS_o, lS_i, use_gpu, use_fp16, device)

                # loss
                E = loss_fn_wrap(Z, T, use_gpu, use_fp16, device)
                '''
                # debug prints
                print("output and loss")
                print(Z.detach().cpu().numpy())
                print(E.detach().cpu().numpy())
                '''
                # compute loss and accuracy
                L = E.detach().cpu().numpy()  # numpy array
                S = Z.detach().cpu().numpy()  # numpy array
                T = T.detach().cpu().numpy()  # numpy array
                mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                A = np.sum((np.round(S, 0) == T).astype(np.uint8)) / mbs

                if not args.inference_only:
                    # scaled error gradient propagation
                    # (where we do not accumulate gradients across mini-batches)
                    optimizer.zero_grad()
                    # backward pass

                    #use apex
                    if (args.enable_amp):
                        with amp.scale_loss(E, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        E.backward()
                    # debug prints (check gradient norm)
                    # for l in mlp.layers:
                    #     if hasattr(l, 'weight'):
                    #          print(l.weight.grad.norm().item())

                    # optimizer
                    optimizer.step()

                t2 = time_wrap(use_gpu)
                total_time += t2 - t1
                total_accu += A
                total_loss += L
                total_iter += 1

                print_tl = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
                print_ts = (
                    (args.test_freq > 0)
                    and (args.data_generation == "dataset")
                    and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                )

                # print time, loss and accuracy
                if print_tl or print_ts:
                    gT = 1000.0 * total_time / total_iter if args.print_time else -1
                    total_time = 0

                    gL = total_loss / total_iter
                    total_loss = 0

                    gA = total_accu / total_iter
                    total_accu = 0

                    str_run_type = "inference" if args.inference_only else "training"
                    print(
                        "Finished {} it {}/{} of epoch {}, ".format(
                            str_run_type, j + 1, nbatches, k
                        )
                        + "{:.2f} ms/it, loss {:.6f}, accuracy {:3.3f} %".format(
                            gT, gL, gA * 100
                        )
                    )
                    total_iter = 0

                # testing
                if print_ts and not args.inference_only:
                    test_accu = 0
                    test_loss = 0

                    if (args.enable_auc):
                        auc_arr_sz = args.mini_batch_size * len(test_loader)
                        auc_y_true = np.zeros(auc_arr_sz)
                        auc_y_score = np.zeros(auc_arr_sz)

                    for jt, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(test_loader):
                        # early exit if nbatches was set by the user and has been exceeded
                        if jt >= nbatches:
                            break

                        t1_test = time_wrap(use_gpu)

                        # forward pass
                        Z_test = dlrm_wrap(
                            X_test, lS_o_test, lS_i_test, use_gpu, use_fp16, device
                        )
                        # loss
                        E_test = loss_fn_wrap(Z_test, T_test, use_gpu,
                                use_fp16, device)

                        # compute loss and accuracy
                        L_test = E_test.detach().cpu().numpy()  # numpy array
                        S_test = Z_test.detach().cpu().numpy()  # numpy array
                        T_test = T_test.detach().cpu().numpy()  # numpy array
                        mbs_test = T_test.shape[
                            0
                        ]  # = args.mini_batch_size except maybe for last
                        A_test = (
                            np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))
                            / mbs_test
                        )

                        t2_test = time_wrap(use_gpu)

                        test_accu += A_test
                        test_loss += L_test

                        if (args.enable_auc):
                            Z_np = S_test.reshape(1, len(S_test))
                            T_np = T_test.reshape(1, len(T_test))
                            b = jt * len(Z_test)
                            e = b + len(Z_test)
                            auc_y_true[b:e] = T_np[:]
                            auc_y_score[b:e] = Z_np[:]
                            del T_np


                    auc = 0.0
                    if (args.enable_auc):
                        auc = roc_auc_score(auc_y_true, auc_y_score)
                        print("AUC = {:.12f}".format(auc))

                    gL_test = test_loss / nbatches_test
                    gA_test = test_accu / nbatches_test

                    is_best = gA_test > best_gA_test
                    if is_best:
                        best_gA_test = gA_test
                        best_gL_test = gL_test
                        best_auc = auc
                        if not (args.save_model == ""):
                            print("Saving model to {}".format(args.save_model))
                            torch.save(
                                {
                                    "epoch": k,
                                    "nepochs": args.nepochs,
                                    "nbatches": nbatches,
                                    "nbatches_test": nbatches_test,
                                    "iter": j + 1,
                                    "state_dict": dlrm.state_dict(),
                                    "train_acc": gA,
                                    "train_loss": gL,
                                    "test_acc": gA_test,
                                    "test_loss": gL_test,
                                    "total_loss": total_loss,
                                    "total_accu": total_accu,
                                    "opt_state_dict": optimizer.state_dict(),
                                    "auc" : auc,
                                },
                                args.save_model,
                            )

                    print(
                        "Testing at - {}/{} of epoch {}, ".format(j + 1, nbatches, 0)
                        + "loss {:.6f}, accuracy {:3.3f} %, best {:3.3f} %, auc {:.12f}".format(
                            gL_test, gA_test * 100, best_gA_test * 100, auc
                        )
                    )
                    print("Best Acc {}, Loss {}, AUC {}".format(best_gA_test, best_gL_test, best_auc))

            k += 1  # nepochs
            #shuffle our dataset
            if (args.avazu_epoch_shuffle):
                train_loader.dataset.shuffle()
            scheduler.step()


    # profiling
    if args.enable_profiling:
        with open("dlrm_s_pytorch.prof", "w") as prof_f:
            prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
            prof.export_chrome_trace("./dlrm_s_pytorch.json")
        # print(prof.key_averages().table(sort_by="cpu_time_total"))

    # plot compute graph
    if args.plot_compute_graph:
        sys.exit(
            "ERROR: Please install pytorchviz package in order to use the"
            + " visualization. Then, uncomment its import above as well as"
            + " three lines below and run the code again."
        )
        # V = Z.mean() if args.inference_only else E
        # dot = make_dot(V, params=dict(dlrm.named_parameters()))
        # dot.render('dlrm_s_pytorch_graph') # write .pdf file

    # test prints
    if not args.inference_only and args.debug_mode:
        print("updated parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())

    # export the model in onnx
    if args.save_onnx:
        with open("dlrm_s_pytorch.onnx", "w+b") as dlrm_pytorch_onnx_file:
            (X, lS_o, lS_i, _) = train_data[0] # get first batch of elements
            torch.onnx._export(
                dlrm, (X, lS_o, lS_i), dlrm_pytorch_onnx_file, verbose=True
            )
        # recover the model back
        dlrm_pytorch_onnx = onnx.load("dlrm_s_pytorch.onnx")
        # check the onnx model
        onnx.checker.check_model(dlrm_pytorch_onnx)
