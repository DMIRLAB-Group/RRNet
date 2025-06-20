import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

import math

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        # nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.normal_(m.weight, 0.0, 0.02)
        # nn.init.normal_(m.weight, 1.5, 0.5)
        nn.init.zeros_(m.bias)


class MLP(nn.Module):
    def __init__(self, cfg, dropout=0.):
        """cfg: [(input_size, output_size, act), ...]"""
        super(MLP, self).__init__()

        density_blocks = []

        # todo: add batchnorm, dropout
        for layer_cfg in cfg:
            density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1]))
            density_blocks.append(nn.Dropout(dropout))
            if layer_cfg[2] == "relu":
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[2] == "leakyrelu":
                density_blocks.append(nn.LeakyReLU(0.2, inplace=True))
            elif layer_cfg[2] == "sigmoid":
                density_blocks.append(nn.Sigmoid())
            elif layer_cfg[2] == "softmax":
                density_blocks.append(nn.Softmax(dim=1))
            else:
                # todo:add more activate function
                pass

        self.mlp = nn.Sequential(*density_blocks)

        self.mlp.apply(init_weights)

    def forward(self, x):
        out = self.mlp(x)
        return out


class Sparse_ps_nn(nn.Module):
    """https://debuggercafe.com/sparse-autoencoders-using-kl-divergence-with-pytorch/"""
    def __init__(self, encoder_model, decoder_model, beta=0.01, type="cla", rho=0.05):
        """
        :param encoder_cfg: param_config of encoder
        :param decoder_cfg: param_config of decoder
        :param z_dim: hidden_size of z
        :param beta: weight of kl_loss
        """
        super(Sparse_ps_nn, self).__init__()

        self.beta = beta
        self.type = type
        self.rho = rho

        self.encoder = encoder_model
        self.decoder = decoder_model

        self.model_children = list(encoder_model.children())[0]


    def forward(self, x, g=None):
        z = self.encoder(x)

        if self.type == "cla":
            y_pred = self.decoder(z)
        else:
            assert g is not None
            y_pred = self.decoder(g=g, x=z)

        return y_pred


    def kl_divergence(self, rho, rho_hat):
        rho_hat = torch.mean(rho_hat, 1)  # sigmoid because we need the probability distributions
        rho = torch.tensor([rho] * len(rho_hat)).cuda()
        return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))

    # define the sparse loss function
    def sparse_loss(self, x):
        values = x
        loss = 0
        for i in range(len(self.model_children) // 3):
            values = self.model_children[i*3+0](values)
            values = self.model_children[i*3+1](values)
            values = self.model_children[i*3+2](values)
            loss += self.kl_divergence(self.rho, values)
        return loss


    def loss_function(self, y, y_pred, x):
        if self.type == "cla":
            y = y.reshape(-1, 1)
            pred_loss = F.binary_cross_entropy(y_pred, y)
            # pred_loss = F.cross_entropy(y_pred, y, reduction="mean")
        elif self.type == "reg":
            pred_loss = - torch.log(y_pred + 1e-5).mean()

        # sparsity
        kl = self.sparse_loss(x)

        loss = pred_loss + self.beta * kl

        return loss, pred_loss, self.beta * kl


class DeepVIB(nn.Module):
    """https://github.com/udeepam/vib/blob/master/vib.ipynb"""
    def __init__(self, encoder_model, decoder_model, z_dim, beta=0.01, type="cla"):
        """
        :param encoder_cfg: param_config of encoder
        :param decoder_cfg: param_config of decoder
        :param z_dim: hidden_size of z
        :param beta: weight of kl_loss
        """
        super(DeepVIB, self).__init__()

        self.z_dim = z_dim
        self.beta = beta
        self.type = type

        self.encoder = encoder_model
        self.decoder = decoder_model

        # self.bn = nn.BatchNorm1d(25)

        # self.bce_loss = nn.BCELoss()


    def reparameter_trick(self, mu, std):
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x, g=None):
        z_param = self.encoder(x)
        mu = z_param[:, :self.z_dim]
        std = z_param[:, self.z_dim:].exp()
        # std = F.softplus(z_param[:, self.z_dim:] - 5, beta=1)


        z = self.reparameter_trick(mu, std)

        if self.type == "cla":
            # z = self.bn(z)
            y_pred = self.decoder(z)
        else:
            assert g is not None
            y_pred = self.decoder(g=g, x=z)

        return y_pred, mu, std

    def loss_function(self, y, y_pred, mu, std):
        if self.type == "cla":
            y = y.reshape(-1, 1)
            pred_loss = F.binary_cross_entropy(y_pred, y)
            # pred_loss = F.cross_entropy(y_pred, y, reduction="mean")
        elif self.type == "reg":
            pred_loss = - torch.log(y_pred + 1e-5).mean()

        # warn: std为0取log导致RuntimeError: CUDA error: device-side assert triggered
        std = std + 1e-5
        kl = (0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)) / y.shape[0]

        loss = pred_loss + self.beta * kl

        return loss, pred_loss, self.beta * kl


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.register_parameter("weight", self.weight)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            self.register_parameter('bias', self.bias)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        #
        # self.weight.data.normal_(0, 0.02)
        # if self.bias is not None:
        #     self.bias.data.normal_(0, 0.02)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_Mine(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(GCN_Mine, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        num = adj.shape[0]
        # diag = torch.diag(torch.cuda.FloatTensor([1 for _ in range(num)]))
        # x = F.relu(self.gc1(x, adj+diag))
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout)
        return x


def comp_grid(y, num_grid):

    # L gives the lower index
    # U gives the upper index
    # inter gives the distance to the lower int

    U = torch.ceil(y * num_grid)
    inter = 1 - (U - y * num_grid)  # 与L的距离
    L = U - 1
    L += (L < 0).int()

    return L.int().tolist(), U.int().tolist(), inter


class Density_Block(nn.Module):
    def __init__(self, num_grid, ind, isbias=1):
        super(Density_Block, self).__init__()
        """
        Assume the variable is bounded by [0,1]
        the output grid: 0, 1/B, 2/B, ..., B/B; output dim = B + 1; num_grid = B
        """
        self.ind = ind
        self.num_grid = num_grid
        self.outd = num_grid + 1

        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)  # 拼接t
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        self.softmax = nn.Softmax(dim=1)

    def forward(self, g, x):
        out = torch.matmul(x, self.weight)
        if self.isbias:
            out += self.bias
        out = self.softmax(out)

        L, U, inter = comp_grid(g, self.num_grid)

        # L_out = out[x1, L]
        # U_out = out[x1, U]
        L_out = out.gather(1, torch.cuda.LongTensor(L))
        U_out = out.gather(1, torch.cuda.LongTensor(U))

        out = L_out + (U_out - L_out) * inter

        return out


if __name__ == "__main__":
    # ====================== deep vib ===========================
    x = torch.randn([256, 64])
    z_dim = 16

    # regression test
    reg_y = torch.randn([256, 1])
    encoder_cfg = [(64, 32, "relu"), (32, 32, "relu"), (32, z_dim * 2, None)]
    reg_decoder_cfg = [(z_dim, 32, "relu"), (32, 32, "relu"), (32, 1, None)]

    dvib_reg = DeepVIB(encoder_cfg, reg_decoder_cfg, z_dim)

    y_pred, mu, std = dvib_reg(x)
    toal_loss, pred_loss, kl_loss = dvib_reg.loss_function(reg_y, y_pred, mu, std, type="reg")

    print(toal_loss.item(), pred_loss.item(), kl_loss.item())

    # classification test
    cla_y = torch.randint(0, 2, [256,]).long()
    cla_decoder_cfg = [(z_dim, 32, "relu"), (32, 32, "relu"), (32, 2, None)]
    dvib_cla = DeepVIB(encoder_cfg, cla_decoder_cfg, z_dim)

    y_pred, mu, std = dvib_cla(x)
    toal_loss, pred_loss, kl_loss  = dvib_cla.loss_function(cla_y, y_pred, mu, std, type="cla")

    print(toal_loss.item(), pred_loss.item(), kl_loss.item())
