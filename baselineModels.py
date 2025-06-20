import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from modules_mine import GCN_Mine
from modules import GCN

class GCN_DECONF(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=3, n_out=2, cuda=True):
        super(GCN_DECONF, self).__init__()

        if cuda:
            self.gc = nn.ModuleList([GraphConvolution(nfeat, nhid)]).cuda()
            for i in range(n_in - 1):
                self.gc.append(GraphConvolution(nhid, nhid).cuda())
        else:
            self.gc = nn.ModuleList([GraphConvolution(nfeat, nhid)])
            for i in range(n_in - 1):
                self.gc.append(GraphConvolution(nhid, nhid))
        
        self.n_in = n_in
        self.n_out = n_out

        if cuda:

            self.out_t00 = [nn.Linear(nhid,nhid).cuda() for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid,nhid).cuda() for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid,1).cuda()
            self.out_t11 = nn.Linear(nhid,1).cuda()

        else:
            self.out_t00 = [nn.Linear(nhid,nhid) for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid,nhid) for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid,1)
            self.out_t11 = nn.Linear(nhid,1)

        self.dropout = dropout

        # a linear layer for propensity prediction
        self.pp = nn.Linear(nhid, 1)

        if cuda:
            self.pp = self.pp.cuda()
        self.pp_act = nn.Sigmoid()

    def forward(self, adj,x, t, Z=None,):

        if Z is None:
            neighbors = torch.sum(adj, 1)
            neighborAverageT = torch.div(torch.matmul(adj, t.reshape(-1)), neighbors)
        else:
            neighborAverageT = Z

        rep = F.relu(self.gc[0](x, adj))
        rep = F.dropout(rep, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep = F.relu(self.gc[i](rep, adj))
            rep = F.dropout(rep, self.dropout, training=self.training)

        for i in range(self.n_out):

            y00 = F.relu(self.out_t00[i](rep))
            y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep))
            y10 = F.dropout(y10, self.dropout, training=self.training)
        
        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)

        y = torch.where(t > 0,y1,y0)

        p1 = self.pp_act(self.pp(rep)).view(-1)

        return p1, -1,y,rep,neighborAverageT



class CFR(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=3, n_out=2, cuda=True):
        super(CFR, self).__init__()

        if cuda:
            self.gc = nn.ModuleList([nn.Linear(nfeat, nhid)]).cuda()
            for i in range(n_in - 1):
                self.gc.append(nn.Linear(nhid,nhid).cuda())
        else:
            self.gc = nn.ModuleList([nn.Linear(nfeat, nhid)])
            for i in range(n_in - 1):
                self.gc.append(nn.Linear(nhid, nhid))
        
        self.n_in = n_in
        self.n_out = n_out

        if cuda:

            self.out_t00 = [nn.Linear(nhid,nhid).cuda() for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid,nhid).cuda() for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid,1).cuda()
            self.out_t11 = nn.Linear(nhid,1).cuda()

        else:
            self.out_t00 = [nn.Linear(nhid,nhid) for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid,nhid) for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid,1)
            self.out_t11 = nn.Linear(nhid,1)

        self.dropout = dropout

        # a linear layer for propensity prediction
        self.pp = nn.Linear(nhid, 1)

        if cuda:
            self.pp = self.pp.cuda()
        self.pp_act = nn.Sigmoid()

    def forward(self, adj,x, t, Z=None,):

        if Z is None:
            neighbors = torch.sum(adj, 1)
            neighborAverageT = torch.div(torch.matmul(adj, t.reshape(-1)), neighbors)
        else:
            neighborAverageT = Z

        rep = F.relu(self.gc[0](x))
        rep = F.dropout(rep, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep = F.relu(self.gc[i](rep))
            rep = F.dropout(rep, self.dropout, training=self.training)

        for i in range(self.n_out):

            y00 = F.relu(self.out_t00[i](rep))
            y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep))
            y10 = F.dropout(y10, self.dropout, training=self.training)
        
        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)

        y = torch.where(t > 0,y1,y0)

        p1 = self.pp_act(self.pp(rep)).view(-1)

        return p1, -1,y,rep,neighborAverageT



class GCN_DECONF_INTERFERENCE(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=3, n_out=2, cuda=True):
        super(GCN_DECONF_INTERFERENCE, self).__init__()

        if cuda:
            self.gc = nn.ModuleList([GraphConvolution(nfeat, nhid)]).cuda()
            for i in range(n_in - 1):
                self.gc.append(GraphConvolution(nhid, nhid).cuda())
        else:
            self.gc = nn.ModuleList([GraphConvolution(nfeat, nhid)])
            for i in range(n_in - 1):
                self.gc.append(GraphConvolution(nhid, nhid))
        
        self.n_in = n_in
        self.n_out = n_out

        if cuda:

            self.out_t00 = [nn.Linear(nhid+1,nhid).cuda() for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid+1,nhid).cuda() for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid,1).cuda()
            self.out_t11 = nn.Linear(nhid,1).cuda()

        else:
            self.out_t00 = [nn.Linear(nhid+1,nhid) for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid+1,nhid) for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid,1)
            self.out_t11 = nn.Linear(nhid,1)

        self.dropout = dropout

        # a linear layer for propensity prediction
        self.pp = nn.Linear(nhid, 1)

        if cuda:
            self.pp = self.pp.cuda()
        self.pp_act = nn.Sigmoid()

    def forward(self, adj,x, t, Z=None,):

        if Z is None:
            neighbors = torch.sum(adj, 1)
            neighborAverageT = torch.div(torch.matmul(adj, t.reshape(-1)), neighbors)
        else:
            neighborAverageT = Z

        rep = F.relu(self.gc[0](x, adj))
        rep = F.dropout(rep, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep = F.relu(self.gc[i](rep, adj))
            rep = F.dropout(rep, self.dropout, training=self.training)

        rep1 = torch.cat((rep, neighborAverageT.reshape(-1, 1)), 1)
        for i in range(self.n_out):

            y00 = F.relu(self.out_t00[i](rep1))
            y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep1))
            y10 = F.dropout(y10, self.dropout, training=self.training)
        
        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)

        y = torch.where(t > 0,y1,y0)

        p1 = self.pp_act(self.pp(rep)).view(-1)

        return p1, -1,y,rep,neighborAverageT



class CFR_INTERFERENCE(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=3, n_out=2, cuda=True):
        super(CFR_INTERFERENCE, self).__init__()

        if cuda:
            self.gc = nn.ModuleList([nn.Linear(nfeat, nhid)]).cuda()
            for i in range(n_in - 1):
                self.gc.append(nn.Linear(nhid,nhid).cuda())
        else:
            self.gc = nn.ModuleList([nn.Linear(nfeat, nhid)])
            for i in range(n_in - 1):
                self.gc.append(nn.Linear(nhid, nhid))
        
        self.n_in = n_in
        self.n_out = n_out

        if cuda:

            self.out_t00 = [nn.Linear(nhid+1,nhid).cuda() for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid+1,nhid).cuda() for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid,1).cuda()
            self.out_t11 = nn.Linear(nhid,1).cuda()

        else:
            self.out_t00 = [nn.Linear(nhid,nhid) for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid,nhid) for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid,1)
            self.out_t11 = nn.Linear(nhid,1)

        self.dropout = dropout

        # a linear layer for propensity prediction
        self.pp = nn.Linear(nhid, 1)

        if cuda:
            self.pp = self.pp.cuda()
        self.pp_act = nn.Sigmoid()

    def forward(self, adj,x, t, Z=None,):

        if Z is None:
            neighbors = torch.sum(adj, 1)
            neighborAverageT = torch.div(torch.matmul(adj, t.reshape(-1)), neighbors)
        else:
            neighborAverageT = Z

        rep = F.relu(self.gc[0](x))
        rep = F.dropout(rep, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep = F.relu(self.gc[i](rep))
            rep = F.dropout(rep, self.dropout, training=self.training)

        rep1 = torch.cat((rep, neighborAverageT.reshape(-1, 1)), 1)
        for i in range(self.n_out):

            y00 = F.relu(self.out_t00[i](rep1))
            y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep1))
            y10 = F.dropout(y10, self.dropout, training=self.training)
        
        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)

        y = torch.where(t > 0,y1,y0)

        p1 = self.pp_act(self.pp(rep)).view(-1)

        return p1, -1,y,rep,neighborAverageT


class CFR_INTERFERENCE_GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=3, n_out=2, cuda=True):
        super(CFR_INTERFERENCE_GCN, self).__init__()

        if cuda:
            self.gc = nn.ModuleList([nn.Linear(nfeat, nhid)]).cuda()
            for i in range(n_in - 1):
                self.gc.append(nn.Linear(nhid, nhid).cuda())
        else:
            self.gc = nn.ModuleList([nn.Linear(nfeat, nhid)])
            for i in range(n_in - 1):
                self.gc.append(nn.Linear(nhid, nhid))

        self.n_in = n_in
        self.n_out = n_out

        if cuda:
            # self.gnn = GCN_Mine(nfeat=nhid, nclass=nhid, dropout=dropout).cuda()
            self.gnn = GCN(nfeat=nhid, nclass=nhid, dropout=dropout).cuda()
            self.out_t00 = [nn.Linear(2*nhid + 1, nhid).cuda() for i in range(n_out)]
            self.out_t10 = [nn.Linear(2*nhid + 1, nhid).cuda() for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid, 1).cuda()
            self.out_t11 = nn.Linear(nhid, 1).cuda()

        else:
            self.out_t00 = [nn.Linear(nhid, nhid) for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid, nhid) for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid, 1)
            self.out_t11 = nn.Linear(nhid, 1)

        self.dropout = dropout

        # a linear layer for propensity prediction
        self.pp = nn.Linear(nhid, 1)

        if cuda:
            self.pp = self.pp.cuda()
        self.pp_act = nn.Sigmoid()

    def forward(self, adj, x, t, Z=None, ):

        if Z is None:
            neighbors = torch.sum(adj, 1)
            neighborAverageT = torch.div(torch.matmul(adj, t.reshape(-1)), neighbors)
        else:
            neighborAverageT = Z

        rep = F.relu(self.gc[0](x))
        rep = F.dropout(rep, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep = F.relu(self.gc[i](rep))
            rep = F.dropout(rep, self.dropout, training=self.training)

        # gcn_rep = self.gnn(x=rep, adj=adj)
        gcn_rep = self.gnn(x=rep * t.view(-1, 1), adj=adj)  # mask

        rep1 = torch.cat((rep, gcn_rep), 1)
        rep1 = torch.cat((rep1, neighborAverageT.reshape(-1, 1)), 1)

        for i in range(self.n_out):
            y00 = F.relu(self.out_t00[i](rep1))
            y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep1))
            y10 = F.dropout(y10, self.dropout, training=self.training)

        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)

        y = torch.where(t > 0, y1, y0)

        p1 = self.pp_act(self.pp(rep)).view(-1)

        return p1, -1, y, rep, neighborAverageT, gcn_rep
