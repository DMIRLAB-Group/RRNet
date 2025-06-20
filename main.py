import argparse
import torch
import time
import utils as utils
import numpy as np
from model import NetEsimator, MyModel
from baselineModels import GCN_DECONF,CFR,GCN_DECONF_INTERFERENCE,CFR_INTERFERENCE,CFR_INTERFERENCE_GCN
from experiment import Experiment
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0,help='Use CUDA training.')
parser.add_argument('--use_cuda', type=bool, default=True,help='Use CUDA training.')
parser.add_argument('--alpha', type=float, default=0.5,help='trade-off of p(t|x).')
parser.add_argument('--gamma', type=float, default=0.5,help='trade-off of p(z|x,t).')
parser.add_argument('--lr', type=float, default=1e-3,help='Initial learning rate.')
parser.add_argument('--lrD', type=float, default=1e-3,help='Initial learning rate of Discriminator.')
parser.add_argument('--lrD_z', type=float, default=1e-3,help='Initial learning rate of Discriminator_z.')
parser.add_argument('--weight_decay', type=float, default=1e-5,help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dstep', type=int, default=10,help='epoch of training discriminator')
parser.add_argument('--d_zstep', type=int, default=10,help='epoch of training discriminator_z')
parser.add_argument('--pstep', type=int, default=1,help='epoch of training')
parser.add_argument('--normy', type=int, default=1)
parser.add_argument('--hidden', type=int, default=32,help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.01,help='Dropout rate (1 - keep probability).')
parser.add_argument('--save_intermediate', type=int, default=0,help='Save training curve and imtermediate embeddings')
parser.add_argument('--printDisc', type=int, default=0,help='Print discriminator result for debug usage')
parser.add_argument('--printDisc_z', type=int, default=0,help='Print discriminator_z result for debug usage')
parser.add_argument('--printPred', type=int, default=1,help='Print encoder-predictor result for debug usage')

parser.add_argument('--seed', type=int, default=123, help='Random seed. RIP KOBE') # 0
parser.add_argument('--epochs', type=int, default=2000,help='Number of epochs to train.') # 500
# [CFR_INTERFERENCE, CFR_INTERFERENCE_GCN, ND_INTERFERENCE, NetEsimator, Mine]
parser.add_argument('--model', type=str, default='Mine',help='Models or baselines')
parser.add_argument("--flipRate", type=float, default=0, help="[0.25, 0.5, 0.75, 1]")
parser.add_argument('--alpha_base', type=float, default=5,help='trade-off of balance for baselines.')

# dataset
# BC_uncon_decom Flickr_uncon_decom
# BC_hete_uncon_decom Flickr_hete_uncon_decom
parser.add_argument('--dataset', type=str, default='BC_uncon_decom') # ["BC","Flickr"]
parser.add_argument('--expID', type=int, default=0)
parser.add_argument('--inter_degree', type=float, default=0.5)
parser.add_argument('--covar_degree', type=float, default=0.5)

# Mine有的参数
parser.add_argument("--beta", type=float, default=1e-3, help="稀疏项的权重")
parser.add_argument("--rho", type=float, default=0.05, help="稀疏项的程度")
parser.add_argument('--lamda', type=float, default=0.1, help="IPM项的权重")
parser.add_argument("--reweighting", type=int, default=1)
parser.add_argument("--max_clamp", type=float, default=10.0)
parser.add_argument("--softmax_w", type=int, default=1)

startTime = time.time()

args = parser.parse_args()
# print(args)
# args.cuda = args.cuda and torch.cuda.is_available()
torch.cuda.set_device(args.cuda)

set_seed(args.seed)

trainA, trainX, trainT,cfTrainT,POTrain,cfPOTrain,\
valA, valX, valT,cfValT,POVal,cfPOVal,\
testA,testX, testT,cfTestT,POTest,cfPOTest,\
train_t1z1,train_t1z0,train_t0z0,train_t0z7,train_t0z2,\
val_t1z1,val_t1z0,val_t0z0,val_t0z7,val_t0z2,\
test_t1z1,test_t1z0,test_t0z0,test_t0z7,test_t0z2 = utils.load_data(args)


if args.model == "NetEsimator":
    model = NetEsimator(Xshape=trainX.shape[1],hidden=args.hidden,dropout=args.dropout)
elif args.model == "ND":
    model = GCN_DECONF(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)
elif args.model == "TARNet":
    model = GCN_DECONF(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)
elif args.model == "CFR":
    model = CFR(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)

elif args.model == "CFR_INTERFERENCE":
    model = CFR_INTERFERENCE(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)
elif args.model == "CFR_INTERFERENCE_GCN":
    model = CFR_INTERFERENCE_GCN(nfeat=trainX.shape[1], nhid=args.hidden, dropout=args.dropout)
elif args.model == "ND_INTERFERENCE":
    model = GCN_DECONF_INTERFERENCE(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)
elif args.model == "TARNet_INTERFERENCE":
    model = GCN_DECONF_INTERFERENCE(nfeat=trainX.shape[1], nhid=args.hidden,dropout=args.dropout)
elif args.model == "Mine": # RRNet
    # model init
    # print(trainX.shape)
    gnn_cfg = [trainX.shape[1], trainX.shape[1], args.dropout]

    # decoder use vcnet
    g_dvib_cfg = {"beta": args.beta, "num_grid": 10, "input_size": 64,
                  "encoder_cfg": [(trainX.shape[1]*2, 64, "sigmoid"), (64, 64, "sigmoid"), (64, 64, "sigmoid")]}

    t_dvib_cfg = {"beta": args.beta,
                  "encoder_cfg": [(trainX.shape[1]*2, 64, "sigmoid"), (64, 64, "sigmoid"), (64, 64, "sigmoid")],
                  "decoder_cfg": [(64, 1, "sigmoid"), ]}

    phi_cfg = [(trainX.shape[1]*2, 64, "relu"), (64, 64, "relu"), (64, 64, "relu")]
    predict_cfg = [(64 + 2, 16, "relu"), (16, 16, "relu"), (16, 1, None)]

    model = MyModel(gnn_cfg=gnn_cfg, g_dvib_cfg=g_dvib_cfg, t_dvib_cfg=t_dvib_cfg, phi_cfg=phi_cfg,
                    predict_cfg=predict_cfg, dropout=args.dropout, rho=args.rho)


exp = Experiment(args,model,trainA, trainX, trainT,cfTrainT,POTrain,cfPOTrain,valA, valX, valT,cfValT,POVal,cfPOVal,testA, testX, testT,cfTestT,POTest,cfPOTest,\
    train_t1z1,train_t1z0,train_t0z0,train_t0z7,train_t0z2,val_t1z1,val_t1z0,val_t0z0,val_t0z7,val_t0z2,test_t1z1,test_t1z0,test_t0z0,test_t0z7,test_t0z2)

"""Train the model"""
exp.train()

"""Moel Predicting"""
exp.predict()


print("Time usage:{:.4f} mins".format((time.time()-startTime)/60))
print ("================================Setting again================================")
print ("Model:{} Dataset:{}, inter_degree:{}, covar_degree:{},  seed:{}, ".format(args.model,args.dataset,args.inter_degree, args.covar_degree,args.seed,))
print ("================================BYE================================\n\n")

