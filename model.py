import argparse
import torch
import pickle as pkl
import torch.nn as nn
import utils as utils
import numpy as np
from modules import GCN, NN, Predictor,Discriminator
from modules_mine import GCN_Mine, MLP, DeepVIB, Density_Block, Sparse_ps_nn

class NetEsimator(nn.Module):

    def __init__(self,Xshape,hidden,dropout):

        super(NetEsimator, self).__init__()
        self.encoder = GCN(nfeat=Xshape, nclass=hidden, dropout=dropout)
        self.predictor = Predictor(input_size=hidden + 2, hidden_size1=hidden, hidden_size2=hidden,output_size=1,dropout=dropout)
        self.discriminator = Discriminator(input_size=hidden,hidden_size1=hidden,hidden_size2=hidden,output_size=1,dropout=dropout)
        self.discriminator_z = Discriminator(input_size=hidden+1,hidden_size1=hidden,hidden_size2=hidden,output_size=1,dropout=dropout)
    

    def forward(self,A,X,T,Z=None):
    
        embeddings = self.encoder(X, A)
        pred_treatment = self.discriminator(embeddings)
        if Z is None:
            neighbors = torch.sum(A, 1)
            neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)
        else:
            neighborAverageT = Z
        embed_treatment = torch.cat((embeddings, T.reshape(-1, 1)), 1) 
        pred_z = self.discriminator_z(embed_treatment)
        embed_treatment_avgT = torch.cat((embed_treatment, neighborAverageT.reshape(-1, 1)), 1)
        pred_outcome0 = self.predictor(embed_treatment_avgT).view(-1)

        return pred_treatment,pred_z,pred_outcome0,embeddings, neighborAverageT



class MyModel(nn.Module):
    def __init__(self, gnn_cfg, g_dvib_cfg, t_dvib_cfg, phi_cfg, predict_cfg, dropout=0., rho=0.05):
        """
        :param gnn_cfg: [input_size, output_size, dropout]
        :param g_dvib_cfg: {"z_dim": z_size,
                            "encoder_cfg":[(input_size, output_size, activate_func),...],
                            "decoder_cfg":[num_grid, input_size]}
        :param t_dvib_cfg: {"z_dim": z_size, "encoder_cfg":[(input_size, output_size, activate_func),...], "decoder_cfg"}
        :param dense_cfg: [(input_size, output_size, activate_func), ...]
        """
        super(MyModel, self).__init__()

        # gnn
        self.gnn = GCN_Mine(nfeat=gnn_cfg[0], nclass=gnn_cfg[1], dropout=gnn_cfg[2])

        # g_dvib
        # assert g_dvib_cfg["encoder_cfg"][-1][1] == 2 * g_dvib_cfg["z_dim"]
        g_encoder = MLP(g_dvib_cfg["encoder_cfg"], dropout=dropout)
        g_decoder = Density_Block(num_grid=g_dvib_cfg["num_grid"], ind=g_dvib_cfg["input_size"])
        self.g_dvib = Sparse_ps_nn(encoder_model=g_encoder, decoder_model=g_decoder,
                                   type="reg", beta=g_dvib_cfg["beta"], rho=rho)

        # t_dvib
        # assert t_dvib_cfg["encoder_cfg"][-1][1] == 2 * t_dvib_cfg["z_dim"]
        t_encoder = MLP(t_dvib_cfg["encoder_cfg"], dropout=dropout)
        t_decoder = MLP(t_dvib_cfg["decoder_cfg"], dropout=dropout)
        self.t_dvib = Sparse_ps_nn(encoder_model=t_encoder, decoder_model=t_decoder,
                                   type="cla", beta=g_dvib_cfg["beta"], rho=rho)

        # fc
        self.phi_nn = MLP(phi_cfg, dropout=dropout)

        # dense_nn
        self.predict_nn = MLP(predict_cfg, dropout=dropout)


    def forward(self, a, x, t, g):
        t, g = t.reshape(-1, 1), g.reshape(-1, 1)

        x_neigh = self.gnn(x, a)

        X = torch.cat((x, x_neigh), dim=1)
        p_g_zt = self.g_dvib(x=X, g=g)
        p_t_z = self.t_dvib(x=X)

        embedding = self.phi_nn(X)

        embed_t_g = torch.cat((embedding, t, g), dim=1)

        y_pred = self.predict_nn(embed_t_g)

        return y_pred, p_t_z, p_g_zt, X, embedding
