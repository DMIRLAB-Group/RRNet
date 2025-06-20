import os
import torch
import pickle as pkl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils as utils
import numpy as np

# ============================ HSIC ================================
def distmat(X):
    """ distance matrix
    """
    r = torch.sum(X*X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X,0,1))
    D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)
    D = torch.abs(D)
    return D

def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(torch.cat([X,Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med=np.mean(Tri)
    if med<1E-2:
        med=1E-2
    return med

def kernelmat(X, sigma):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    dim = int(X.size()[1]) * 1.0
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])
    Dxx = distmat(X)

    if sigma:
        variance = 2. * sigma * sigma * X.size()[1]
        Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
        # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
    else:
        try:
            sx = sigma_estimation(X, X)
            Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
        except RuntimeError as e:
            raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                sx, torch.max(X), torch.min(X)))

    Kxc = torch.mm(Kx, H)

    return Kxc

def hsic_regular(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """
    """
    Kxc = kernelmat(x, sigma)
    Kyc = kernelmat(y, sigma)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy

def hsic_normalized(x, y, sigma=None, use_cuda=True, to_numpy=True):
    """
    """
    m = int(x.size()[0])
    Pxy = hsic_regular(x, y, sigma, use_cuda)
    Px = torch.sqrt(hsic_regular(x, x, sigma, use_cuda))
    Py = torch.sqrt(hsic_regular(y, y, sigma, use_cuda))
    thehsic = Pxy/(Px*Py)
    return thehsic


# =========================== wasserstein ========================
def wasserstein(x, y, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=True):
    """return W dist between x and y"""
    '''distance matrix M'''
    nx = x.shape[0]
    ny = y.shape[0]

    x = x.squeeze()
    y = y.squeeze()

    #    pdist = torch.nn.PairwiseDistance(p=2)

    M = pdist(x, y)  # distance_matrix(x,y,p=2)

    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, 10.0 / (nx * ny))
    delta = torch.max(M_drop).detach()
    eff_lam = (lam / M_mean).detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta * torch.ones(M[0:1, :].shape).cuda()
    col = torch.cat([delta * torch.ones(M[:, 0:1].shape).cuda(), torch.zeros((1, 1)).cuda()], 0)
    # if cuda:
    #     row = row.cuda()
    #     col = col.cuda()
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)

    '''compute marginal'''
    a = torch.cat([p * torch.ones((nx, 1)) / nx, (1 - p) * torch.ones((1, 1))], 0)
    b = torch.cat([(1 - p) * torch.ones((ny, 1)) / ny, p * torch.ones((1, 1))], 0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1) * 1e-6
    if cuda:
        temp_term = temp_term.cuda()
        a = a.cuda()
        b = b.cuda()
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K / a

    u = a

    for i in range(its):
        u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
        # if cuda:
        #     u = u.cuda()
    v = b / (torch.t(torch.t(u).matmul(K)))
    # if cuda:
    #     v = v.cuda()

    upper_t = u * (torch.t(v) * K).detach()

    E = upper_t * Mt
    D = 2 * torch.sum(E)

    # if cuda:
    #     D = D.cuda()

    return D, Mlam


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

# ============================== MMD ============================
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5., fix_sigma=5.):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


class Experiment():

    def __init__(self,args,model,trainA, trainX, trainT,cfTrainT,POTrain,cfPOTrain,valA, valX, valT,cfValT,POVal,cfPOVal,testA, testX, testT,cfTestT,POTest,cfPOTest,
        train_t1z1,train_t1z0,train_t0z0,train_t0z7,train_t0z2,val_t1z1,val_t1z0,val_t0z0,val_t0z7,val_t0z2,test_t1z1,test_t1z0,test_t0z0,test_t0z7,test_t0z2):
        super(Experiment, self).__init__()

        self.args = args
        self.model = model
        if self.args.model=="NetEsimator":
            self.optimizerD = optim.Adam([{'params':self.model.discriminator.parameters()}],lr=self.args.lrD, weight_decay=self.args.weight_decay)
            self.optimizerD_z = optim.Adam([{'params':self.model.discriminator_z.parameters()}],lr=self.args.lrD_z, weight_decay=self.args.weight_decay)
            self.optimizerP = optim.Adam([{'params':self.model.encoder.parameters()},{'params':self.model.predictor.parameters()}],lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.model == "Mine":
            self.optimizer_g = optim.Adam([{'params': self.model.g_dvib.parameters()}], lr=args.lr,
                                          weight_decay=args.weight_decay)
            self.optimizer_t = optim.Adam([{'params': self.model.t_dvib.parameters()}], lr=args.lr,
                                          weight_decay=args.weight_decay)
            self.optimizer_gnn_y = optim.Adam(
                [{'params': self.model.gnn.parameters()}, {'params': self.model.phi_nn.parameters()},
                 {'params': self.model.predict_nn.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizerB = optim.Adam(self.model.parameters(),lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.use_cuda:
            self.model = self.model.cuda()
        # print ("================================Model================================")
        # print(self.model)

        # self.Tensor = torch.cuda.FloatTensor if self.args.cuda != 0 else torch.FloatTensor
        self.Tensor = torch.cuda.FloatTensor

        self.trainA = trainA.cuda()
        self.trainX = trainX.cuda()
        self.trainT = trainT.cuda()
        self.trainZ = self.compute_z(self.trainT,self.trainA)
        self.cfTrainT = cfTrainT.cuda()
        self.POTrain = POTrain.cuda()
        self.cfPOTrain = cfPOTrain.cuda()

        self.valA = valA.cuda()
        self.valX = valX.cuda()
        self.valT = valT.cuda()
        self.valZ = self.compute_z(self.valT,self.valA)
        self.cfValT = cfValT.cuda()
        self.POVal = POVal.cuda()
        self.cfPOVal = cfPOVal.cuda()

        self.testA = testA.cuda()
        self.testX = testX.cuda()
        self.testT = testT.cuda()
        self.testZ = self.compute_z(self.testT,self.testA)
        self.cfTestT = cfTestT.cuda()
        self.POTest = POTest.cuda()
        self.cfPOTest = cfPOTest.cuda()

        self.z_1 = 0.7
        self.z_2 = 0.2
        self.train_t1z1 = self.Tensor(train_t1z1)
        self.train_t1z0 = self.Tensor(train_t1z0)
        self.train_t0z0 = self.Tensor(train_t0z0)
        self.train_t0z7 = self.Tensor(train_t0z7)
        self.train_t0z2 = self.Tensor(train_t0z2)

        self.val_t1z1 = self.Tensor(val_t1z1)
        self.val_t1z0 = self.Tensor(val_t1z0)
        self.val_t0z0 = self.Tensor(val_t0z0)
        self.val_t0z7 = self.Tensor(val_t0z7)
        self.val_t0z2 = self.Tensor(val_t0z2)

        self.test_t1z1 = self.Tensor(test_t1z1)
        self.test_t1z0 = self.Tensor(test_t1z0)
        self.test_t0z0 = self.Tensor(test_t0z0)
        self.test_t0z7 = self.Tensor(test_t0z7)
        self.test_t0z2 = self.Tensor(test_t0z2)


        """PO normalization if any"""
        self.YFTrain,self.YCFTrain = utils.PO_normalize(self.args.normy,self.POTrain,self.POTrain,self.cfPOTrain)
        self.YFVal,self.YCFVal = utils.PO_normalize(self.args.normy,self.POTrain,self.POVal,self.cfPOVal)
        self.YFTest ,self.YCFTest = utils.PO_normalize(self.args.normy,self.POTrain,self.POTest,self.cfPOTest)

        self.loss = nn.MSELoss()
        self.d_zLoss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.peheLoss = nn.MSELoss()
        # kernel_mul = 1, kernel_num = 45, fix_sigma=1
        self.mmd_loss = MMD_loss(kernel_mul = 1, kernel_num = 45, fix_sigma=1)

        
        self.alpha = self.Tensor([self.args.alpha])
        self.gamma = self.Tensor([self.args.gamma])
        self.alpha_base = self.Tensor([self.args.alpha_base])
        if self.args.use_cuda:
            torch.cuda.manual_seed(self.args.seed)
            self.loss = self.loss.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.peheLoss = self.peheLoss.cuda()
            self.mmd_loss = self.mmd_loss.cuda()

        self.lossTrain = []
        self.lossVal = []
        self.lossTest = []
        self.lossCFTrain = []
        self.lossCFVal= []
        self.lossCFTest= []

        self.dissTrain = []
        self.dissVal = []
        self.dissTrainHalf = []
        self.dissValHalf = []
        self.diss_zTrain = []
        self.diss_zVal = []
        self.diss_zTrainHalf = []
        self.diss_zValHalf = []

        self.labelTrain = []
        self.labelVal = []
        self.labelTest = []
        self.labelTrainCF = []
        self.labelValCF = []
        self.labelTestCF = []

        self.predT = []
        self.labelT = []

    def loss_func_y(self, y, y_pred, w):
        y = y.reshape(-1, 1)
        loss = (w * (y - y_pred)**2).mean()
        return loss

    def loss_func_mmd(self, x, y):
        loss = self.mmd_loss(x, y)
        return loss

    def loss_func_hsic(self, x, y):
        loss = hsic_normalized(x, y)
        return loss

    def loss_func_wass(self, x, y):
        loss, _ = wasserstein(x, y)
        return loss

    def get_peheLoss(self,y1pred,y0pred,y1gt,y0gt):
        pred = y1pred - y0pred
        gt = y1gt - y0gt
        return torch.sqrt(self.peheLoss(pred,gt))
    
    def compute_z(self,T,A):
        # print ("A has identity?: {}".format(not (A[0][0]==0 and A[24][24]==0 and A[8][8]==0)))
        neighbors = torch.sum(A, 1)
        neighborAverageT = torch.div(torch.matmul(A, T.reshape(-1)), neighbors)
        return neighborAverageT




    def train_one_step_discriminator(self,A,X,T):

        self.model.train()
        self.optimizerD.zero_grad()
        pred_treatmentTrain,_, _, _,_ = self.model(A,X,T)
        discLoss = self.bce_loss(pred_treatmentTrain.reshape(-1),T)
        num = pred_treatmentTrain.shape[0]
        target05 = [0.5 for _ in range(num)]
        discLosshalf = self.loss(pred_treatmentTrain.reshape(-1), self.Tensor(target05))
        discLoss.backward()
        self.optimizerD.step()

        return discLoss,discLosshalf


    def eval_one_step_discriminator(self,A,X,T):

        self.model.eval()
        pred_treatment,_,_,_,_ = self.model(A,X,T)
        discLossWatch = self.bce_loss(pred_treatment.reshape(-1), T)
        num = pred_treatment.shape[0]
        target05 = [0.5 for _ in range(num)]
        discLosshalf = self.loss(pred_treatment.reshape(-1), self.Tensor(target05))

        return discLossWatch,discLosshalf,pred_treatment,T
        

    def train_discriminator(self,epoch):
        
        for ds in range(self.args.dstep):
            discLoss,discLossTrainhalf = self.train_one_step_discriminator(self.trainA, self.trainX, self.trainT)
            discLossVal,discLossValhalf,_,_ = self.eval_one_step_discriminator(self.valA,self.valX,self.valT)
            discLossTest,discLossTesthalf,_,_ = self.eval_one_step_discriminator(self.testA,self.testX,self.testT)

            if ds == self.args.dstep-1:
                if self.args.printDisc:
                    print('d_Epoch: {:04d}'.format(epoch + 1),
                        'dLoss:{:05f}'.format(discLoss),
                        'dLossVal:{:05f}'.format(discLossVal),
                        'dLossTest:{:05f}'.format(discLossTest),
                        'dLoss0.5:{:05f}'.format(discLossTrainhalf),
                        'dLossVal0.5:{:05f}'.format(discLossValhalf),
                        'dLossTest0.5:{:05f}'.format(discLossTesthalf),
                        )
                self.dissTrain.append(discLoss.detach().cpu().numpy())
                self.dissVal.append(discLossVal.detach().cpu().numpy())
                self.dissTrainHalf.append(discLossTrainhalf.detach().cpu().numpy())
                self.dissValHalf.append(discLossValhalf.detach().cpu().numpy())


    def train_one_step_discriminator_z(self,A,X,T):

        self.model.train()
        self.optimizerD_z.zero_grad()
        _,pred_zTrain,_, _,labelZ = self.model(A,X,T)
        discLoss_z = self.d_zLoss(pred_zTrain.reshape(-1),labelZ)
        num = pred_zTrain.shape[0]
        target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
        discLosshalf_z = self.loss(pred_zTrain.reshape(-1), self.Tensor(target))
        discLoss_z.backward()
        self.optimizerD_z.step()

        return discLoss_z,discLosshalf_z


    def eval_one_step_discriminator_z(self,A,X,T):

        self.model.eval()
        _,pred_z,_,_,labelZ = self.model(A,X,T)
        discLossWatch = self.d_zLoss(pred_z.reshape(-1), labelZ)
        num = pred_z.shape[0]
        target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
        discLosshalf = self.loss(pred_z.reshape(-1), self.Tensor(target))

        return discLossWatch,discLosshalf,pred_z,labelZ


    def train_discriminator_z(self,epoch):
        
        for dzs in range(self.args.d_zstep):
            discLoss_z,discLoss_zTrainRandom = self.train_one_step_discriminator_z(self.trainA, self.trainX, self.trainT)
            discLoss_zVal,discLoss_zValRandom,_,_ = self.eval_one_step_discriminator_z(self.valA,self.valX,self.valT)
            discLoss_zTest,discLoss_zTestRandom,_,_ = self.eval_one_step_discriminator_z(self.testA,self.testX,self.testT)

            if dzs == self.args.d_zstep-1:
                if self.args.printDisc_z:
                    print('d_Epoch: {:04d}'.format(epoch + 1),
                        'd_zLoss:{:05f}'.format(discLoss_z),
                        'd_zLossVal:{:05f}'.format(discLoss_zVal),
                        'd_zLossTest:{:05f}'.format(discLoss_zTest),
                        'd_zLRanTrain:{:05f}'.format(discLoss_zTrainRandom),
                        'd_zLRanVal:{:05f}'.format(discLoss_zValRandom),
                        'd_zLRanTest:{:05f}'.format(discLoss_zTestRandom),
                        )
                self.diss_zTrain.append(discLoss_z.detach().cpu().numpy())
                self.diss_zVal.append(discLoss_zVal.detach().cpu().numpy())
                self.diss_zTrainHalf.append(discLoss_zTrainRandom.detach().cpu().numpy())
                self.diss_zValHalf.append(discLoss_zValRandom.detach().cpu().numpy())


    def train_one_step_encoder_predictor(self,A,X,T,Y):
        
        if self.args.model == "NetEsimator":
            self.model.zero_grad()
            self.model.train()
            self.optimizerP.zero_grad()
            pred_treatmentTrain, pred_zTrain,pred_outcomeTrain,_,_ = self.model(A,X,T)

            pLoss = self.loss(pred_outcomeTrain.reshape(-1), Y)

            # make embedding learned by encoder to predict T inaccurately, i.e., loss T's information
            num = pred_treatmentTrain.shape[0]
            target05 = [0.5 for _ in range(num)]
            dLoss = self.loss(pred_treatmentTrain.reshape(-1), self.Tensor(target05))

            # make embedding learned by encoder to predict Z inaccurately, i.e., loss Z's information
            num = pred_zTrain.shape[0]
            target = self.Tensor(np.random.uniform(low=0.0, high=1.0, size=num))
            d_zLoss = self.d_zLoss(pred_zTrain.reshape(-1), target)

            loss_train = pLoss + dLoss*self.alpha + d_zLoss*self.gamma
            loss_train.backward()
            self.optimizerP.step()

        elif self.args.model == "Mine":
            self.model.zero_grad()
            self.model.train()
            self.optimizer_gnn_y.zero_grad()
            G = self.compute_z(T, A).reshape(-1, 1)

            y_pred, p_t_z, p_g_zt, X, embedding = self.model(x=X, a=A, t=T, g=G)

            if self.args.reweighting == 1:
                T = T.reshape(-1, 1)
                p_t_z = T * p_t_z + (1 - T) * (1 - p_t_z)
                w_t = 1. / (p_t_z + 1e-3)
                w_t = torch.clamp(w_t, max=self.args.max_clamp)
                # print("w_t: max:{:.4f}, min:{:.4f}, mean:{:.4f}".format(w_t.max().item(), w_t.min().item(),
                #                                                         w_t.mean().item()))

                w_g = 1. / (p_g_zt + 1e-3)
                w_g = torch.clamp(w_g, max=self.args.max_clamp)
                # print("w_g: max:{:.4f}, min:{:.4f}, mean:{:.4f}".format(w_g.max().item(), w_g.min().item(),
                #                                                         w_g.mean().item()))

                w = w_t * w_g
                # print("before w: max:{:.4f}, min:{:.4f}, mean:{:.4f}".format(w.max().item(), w.min().item(),
                #                                                         w.mean().item()))

                if self.args.softmax_w == 1:
                    w = torch.sigmoid(w) * 2
                    w = torch.exp(w) / torch.exp(w).sum() * w.shape[0]

                    # w = w / w.sum() * w.shape[0]
                # print("after w: max:{:.4f}, min:{:.4f}, mean:{:.4f}, 0_num:{}".format(w.max().item(), w.min().item(),
                #                                                              w.mean().item(), (w==0).sum().item()))
                # print("-----------------------------------------------------------")
            else:
                T = T.reshape(-1, 1)
                w = 1.

            pLoss = self.loss_func_y(Y, y_pred, w)

            ## ------------------- after Wasserstein(repre) -----------------------
            treatment_pair = torch.cat((T, G), dim=1)
            permute_treatment_pair = treatment_pair[torch.randperm(treatment_pair.size(0))]
            weighted_embedding = w * embedding
            x = torch.cat((embedding, permute_treatment_pair), dim=1)
            y = torch.cat((weighted_embedding, treatment_pair), dim=1)
            dLoss = self.loss_func_wass(x, y)    # wass

            loss_train = pLoss + self.args.lamda * dLoss
            d_zLoss = self.Tensor([-1])
            loss_train.backward()
            self.optimizer_gnn_y.step()


        else:
            self.model.zero_grad()
            self.model.train()
            self.optimizerB.zero_grad()
            if self.args.model in set(["TARNet","TARNet_INTERFERENCE"]):
                _, _, pred_outcomeTrain, rep, _ = self.model(A, X, T)
                pLoss = self.loss(pred_outcomeTrain.reshape(-1), Y)
                loss_train = pLoss
                dLoss = self.Tensor([0])
            elif self.args.model in set(["CFR_INTERFERENCE_GCN"]):
                _, _, pred_outcomeTrain, rep, _, gcn_rep = self.model(A, X, T)
                pLoss = self.loss(pred_outcomeTrain.reshape(-1), Y)
                dLoss = self.loss_func_hsic(rep, T.reshape(-1, 1))
                loss_train = pLoss + self.alpha_base * dLoss

                # d_zLoss = self.loss_func_hsic(gcn_rep, T.reshape(-1, 1))
                # loss_train = pLoss + self.alpha_base * dLoss + self.alpha_base * d_zLoss
            else:
                _, _, pred_outcomeTrain, rep, _, = self.model(A, X, T)
                pLoss = self.loss(pred_outcomeTrain.reshape(-1), Y)
                rep_t1, rep_t0 = rep[(T > 0).nonzero()], rep[(T < 1).nonzero()]
                dLoss,_= utils.wasserstein(rep_t1, rep_t0, cuda=self.args.use_cuda)
                loss_train = pLoss+self.alpha_base*dLoss
            d_zLoss = self.Tensor([-1])
            loss_train.backward()
            self.optimizerB.step()


        return loss_train,pLoss,dLoss,d_zLoss


    def get_ateLoss(self, y1pred, y0pred, y1gt, y0gt):
        pred = y1pred - y0pred
        gt = y1gt - y0gt
        return torch.abs(torch.mean(pred) - torch.mean(gt))
    
    def compute_effect_pehe(self,A,X,gt_t1z1,gt_t1z0,gt_t0z7,gt_t0z2,gt_t0z0):
            
        num = X.shape[0]
        z_1s = self.Tensor(np.ones(num))
        z_0s = self.Tensor(np.zeros(num))
        z_07s = self.Tensor(np.zeros(num)+self.z_1)
        z_02s = self.Tensor(np.zeros(num)+self.z_2)
        t_1s = self.Tensor(np.ones(num))
        t_0s = self.Tensor(np.zeros(num))

        if self.args.model == "Mine":
            pred_outcome_t1z1, _, _, _, _ = self.model(A,X,t_1s,z_1s)
            pred_outcome_t1z0, _, _, _, _ = self.model(A,X,t_1s,z_0s)
            pred_outcome_t0z0, _, _, _, _ = self.model(A,X,t_0s,z_0s)
            pred_outcome_t0z7, _, _, _, _ = self.model(A,X,t_0s,z_07s)
            pred_outcome_t0z2, _, _, _, _ = self.model(A,X,t_0s,z_02s)
        elif self.args.model == "CFR_INTERFERENCE_GCN":
            _, _, pred_outcome_t1z1, _, _, _ = self.model(A, X, t_1s, z_1s)
            _, _, pred_outcome_t1z0, _, _, _ = self.model(A, X, t_1s, z_0s)
            _, _, pred_outcome_t0z0, _, _, _ = self.model(A, X, t_0s, z_0s)
            _, _, pred_outcome_t0z7, _, _, _ = self.model(A, X, t_0s, z_07s)
            _, _, pred_outcome_t0z2, _, _, _ = self.model(A, X, t_0s, z_02s)
        else:
            _, _,pred_outcome_t1z1,_,_ = self.model(A,X,t_1s,z_1s)
            _, _,pred_outcome_t1z0,_,_ = self.model(A,X,t_1s,z_0s)
            _, _,pred_outcome_t0z0,_,_ = self.model(A,X,t_0s,z_0s)
            _, _,pred_outcome_t0z7,_,_ = self.model(A,X,t_0s,z_07s)
            _, _,pred_outcome_t0z2,_,_ = self.model(A,X,t_0s,z_02s)

        pred_outcome_t1z1 = utils.PO_normalize_recover(self.args.normy,self.POTrain,pred_outcome_t1z1).reshape(-1,)
        pred_outcome_t1z0 = utils.PO_normalize_recover(self.args.normy,self.POTrain,pred_outcome_t1z0).reshape(-1,)
        pred_outcome_t0z0 = utils.PO_normalize_recover(self.args.normy,self.POTrain,pred_outcome_t0z0).reshape(-1,)
        pred_outcome_t0z7 = utils.PO_normalize_recover(self.args.normy,self.POTrain,pred_outcome_t0z7).reshape(-1,)
        pred_outcome_t0z2 = utils.PO_normalize_recover(self.args.normy,self.POTrain,pred_outcome_t0z2).reshape(-1,)

        individual_effect = self.get_peheLoss(pred_outcome_t1z0,pred_outcome_t0z0,gt_t1z0,gt_t0z0)
        peer_effect = self.get_peheLoss(pred_outcome_t0z7,pred_outcome_t0z2,gt_t0z7,gt_t0z2)
        total_effect = self.get_peheLoss(pred_outcome_t1z1,pred_outcome_t0z0,gt_t1z1,gt_t0z0)

        ate_individual = self.get_ateLoss(pred_outcome_t1z0, pred_outcome_t0z0, gt_t1z0, gt_t0z0)
        ate_peer = self.get_ateLoss(pred_outcome_t0z7, pred_outcome_t0z2, gt_t0z7, gt_t0z2)
        ate_total = self.get_ateLoss(pred_outcome_t1z1, pred_outcome_t0z0, gt_t1z1, gt_t0z0)

        return individual_effect,peer_effect,total_effect, ate_individual, ate_peer, ate_total


    
    def train_encoder_predictor(self,epoch):
        
        for _ in range(self.args.pstep):
           loss_train,pLoss_train,dLoss_train,d_zLoss_train = self.train_one_step_encoder_predictor(self.trainA, self.trainX, self.trainT,self.YFTrain)

           self.lossTrain.append(loss_train.cpu().detach().numpy())
           # self.lossVal.append(loss_val.cpu().detach().numpy())
           
           """CHECK CF"""
           with torch.no_grad():
               cfLossTrain, cfPred_train, YCFTrainO,_,_ = self.one_step_predict(self.trainA, self.trainX, self.cfTrainT, self.YCFTrain)
               cfLossTest, cfPred_test, YCFTestO,_,_ = self.one_step_predict(self.testA, self.testX, self.cfTestT,self.YCFTest)
               # cfloss_train,cfPLoss_train,cfDLoss_train,cfD_zLoss_train = self.eval_one_step_encoder_predictor(self.trainA, self.trainX, self.cfTrainT,self.YCFTrain)
               # cfloss_val,cfPLoss_val,cfDLoss_val,cfD_zLoss_val = self.eval_one_step_encoder_predictor(self.valA, self.valX, self.cfValT,self.YCFVal)
               # self.lossCFTrain.append(cfloss_train.cpu().detach().numpy())
               # self.lossCFVal.append(cfloss_val.cpu().detach().numpy())

               individual_effect_train,peer_effect_train,total_effect_train, \
               ate_individual_train, ate_peer_train, ate_total_train = \
                   self.compute_effect_pehe(self.trainA, self.trainX,self.train_t1z1,self.train_t1z0,self.train_t0z7,self.train_t0z2,self.train_t0z0)

               individual_effect_test,peer_effect_test,total_effect_test, \
               ate_individual_test, ate_peer_test, ate_total_test = \
                   self.compute_effect_pehe(self.testA, self.testX,self.test_t1z1,self.test_t1z0,self.test_t0z7,self.test_t0z2,self.test_t0z0)

        if self.args.model in set(["ND_INTERFERENCE", "CFR_INTERFERENCE", "CFR_INTERFERENCE_GCN"]) :
            iter = 20
        else:
            iter = 25

        if self.args.printPred and epoch % iter == 0:
            print('p_Epoch: {:04d}'.format(epoch),
                  'loss_train:{:.4f}'.format(loss_train.item()),
                'pLossTrain:{:.4f}'.format(pLoss_train.item()),
                # 'pLossVal:{:.4f}'.format(pLoss_val.item()),
                # 'dLossTrain:{:.4f}'.format(dLoss_train.item()),
                # 'dLossVal:{:.4f}'.format(dLoss_val.item()),
                # 'd_zLossTrain:{:.4f}'.format(d_zLoss_train.item()),
                # 'd_zLossVal:{:.4f}'.format(d_zLoss_val.item()),
                
                # 'CFpLossTrain:{:.4f}'.format(cfLossTrain.item()),
                # 'CFpLossTest:{:.4f}'.format(cfLossTest.item()),
                # 'CFpLossVal:{:.4f}'.format(cfPLoss_val.item()),
                # 'CFdLossTrain:{:.4f}'.format(cfDLoss_train.item()),
                # 'CFdLossVal:{:.4f}'.format(cfDLoss_val.item()),
                # 'CFd_zLossTrain:{:.4f}'.format(cfD_zLoss_train.item()),
                # 'CFd_zLossVal:{:.4f}'.format(cfD_zLoss_val.item()),

                'iE_train:{:.4f}'.format(individual_effect_train.item()),
                'PE_train:{:.4f}'.format(peer_effect_train.item()),
                'TE_train:{:.4f}'.format(total_effect_train.item()),
                'iE_test:{:.4f}'.format(individual_effect_test.item()),
                'PE_test:{:.4f}'.format(peer_effect_test.item()),
                'TE_test:{:.4f}'.format(total_effect_test.item()),

                # 'AiE_train:{:.4f}'.format(ate_individual_train.item()),
                # 'APE_train:{:.4f}'.format(ate_peer_train.item()),
                # 'ATE_train:{:.4f}'.format(ate_total_train.item()),
                # 'AiE_test:{:.4f}'.format(ate_individual_test.item()),
                # 'APE_test:{:.4f}'.format(ate_peer_test.item()),
                # 'ATE_test:{:.4f}'.format(ate_total_test.item()),

                )

    def train_one_step_dvib_g(self, epoch):
        # print("====================== loss_g ========================")
        loss_list = []
        for i in range(self.args.d_zstep):
            self.model.train()
            self.optimizer_g.zero_grad()
            G = self.compute_z(self.trainT, self.trainA).reshape(-1, 1)
            _, _, p_g_z, X, embedding = self.model(x=self.trainX, a=self.trainA, t=self.trainT, g=G)
            total_loss, pred_loss, kl_loss = self.model.g_dvib.loss_function(y=G, y_pred=p_g_z, x=X)
            total_loss.backward()
            self.optimizer_g.step()

            loss_list.append(pred_loss.item())
            # print("epoch {}: {}".format(i, pred_loss.item()))

        return np.array(loss_list).mean()


    def train_one_step_dvib_t(self, epoch):
        pred_loss_list = []
        # print("====================== loss_t ========================")
        for i in range(self.args.dstep):
            self.model.train()
            self.optimizer_t.zero_grad()
            G = self.compute_z(self.trainT, self.trainA).reshape(-1, 1)
            _, p_t_z,  _, X, embedding = self.model(x=self.trainX, a=self.trainA, t=self.trainT, g=G)
            total_loss, pred_loss, kl_loss = self.model.t_dvib.loss_function(y=self.trainT, y_pred=p_t_z, x=X)
            total_loss.backward()
            self.optimizer_t.step()

            pred_loss_list.append(pred_loss.item())
            # print("epoch {}: {}".format(i, pred_loss.item()))

        return np.array(pred_loss_list).mean()

    def train(self):
        print ("================================Training Start================================")

        if self.args.model == "NetEsimator":
            print ("******************NetEsimator******************")
            for epoch in range(self.args.epochs):
                self.train_discriminator(epoch)
                self.train_discriminator_z(epoch)
                self.train_encoder_predictor(epoch)
        elif self.args.model == "Mine":
            print("******************Mine******************")
            for epoch in range(self.args.epochs):
                loss_g = self.train_one_step_dvib_g(epoch)
                loss_z = self.train_one_step_dvib_t(epoch)
                self.train_encoder_predictor(epoch)

        else:
            print ("******************Baselines:{}******************".format(self.args.model))
            for epoch in range(self.args.epochs):
                self.train_encoder_predictor(epoch)
    

    def one_step_predict(self,A,X,T,Y):
        self.model.eval()
        if self.args.model == "Mine":
            G = self.compute_z(T, A).reshape(-1, 1)
            pred_outcome, p_t_z, p_g_zt, X, embedding = self.model(x=X, a=A, t=T, g=G)

            save_p_t = p_t_z.clone()

            # need compare dLoss
            if self.args.reweighting == 1:
                T = T.reshape(-1, 1)
                p_t_z = T * p_t_z + (1 - T) * (1 - p_t_z)
                w_t = 1. / (p_t_z + 1e-3)
                w_t = torch.clamp(w_t, max=self.args.max_clamp)
                w_g = 1. / (p_g_zt + 1e-3)
                w_g = torch.clamp(w_g, max=self.args.max_clamp)
                w = w_t * w_g
                if self.args.softmax_w == 1:
                    w = torch.sigmoid(w) * 2
                    w = torch.exp(w) / torch.exp(w).sum() * w.shape[0]

                if self.args.lamda == 0:
                    # only reweighting
                    treatment_pair = torch.cat((T, G), dim=1)
                    permute_treatment_pair = treatment_pair[torch.randperm(treatment_pair.size(0))]
                    weighted_embedding = w * embedding
                    x = torch.cat((embedding, permute_treatment_pair), dim=1)
                    y = torch.cat((weighted_embedding, treatment_pair), dim=1)
                    dLoss = self.loss_func_hsic(x, y)
                else:
                    dLoss = self.Tensor([0])

            # needn't compare dLoss
            else:
                if self.args.lamda == 0:
                    # None
                    treatment_pair = torch.cat((T.reshape(-1, 1), G), dim=1)
                    permute_treatment_pair = treatment_pair[torch.randperm(treatment_pair.size(0))]
                    x = torch.cat((X, permute_treatment_pair), dim=1)
                    y = torch.cat((X, treatment_pair), dim=1)
                    dLoss = self.loss_func_hsic(x, y)
                else:
                    # only repre
                    treatment_pair = torch.cat((T.reshape(-1, 1), G), dim=1)
                    permute_treatment_pair = treatment_pair[torch.randperm(treatment_pair.size(0))]
                    x = torch.cat((embedding, permute_treatment_pair), dim=1)
                    y = torch.cat((embedding, treatment_pair), dim=1)
                    dLoss = self.loss_func_hsic(x, y)
        elif self.args.model == "CFR_INTERFERENCE_GCN":
            pred_treatment, _, pred_outcome, _, _, _ = self.model(A, X, T)
            dLoss = self.Tensor([0])
            save_p_t = self.Tensor([0] * pred_treatment.shape[0])
        else:
            pred_treatment, _,pred_outcome,_,_ = self.model(A,X,T)
            dLoss = self.Tensor([0])
            save_p_t = self.Tensor([0] * pred_treatment.shape[0])

        pred_outcome = utils.PO_normalize_recover(self.args.normy,self.POTrain,pred_outcome)
        Y = utils.PO_normalize_recover(self.args.normy,self.POTrain,Y)
        pLoss = torch.sqrt(self.loss(pred_outcome.reshape(-1), Y))
        
        return pLoss,pred_outcome,Y, dLoss, save_p_t
        

    def predict(self):
        print ("================================Predicting================================")
        factualLossTrain,pred_train,YFTrainO,dLoss_train,save_p_t = self.one_step_predict(self.trainA,self.trainX,self.trainT,self.YFTrain)
        factualLossVal,pred_val,YFValO,_,_ = self.one_step_predict(self.valA,self.valX,self.valT,self.YFVal)
        factualLossTest,pred_test,YFTestO,dLoss_test,_ = self.one_step_predict(self.testA,self.testX,self.testT,self.YFTest)

        cfLossTrain,cfPred_train,YCFTrainO,_,_ = self.one_step_predict(self.trainA,self.trainX,self.cfTrainT,self.YCFTrain)
        cfLossVal,cfPred_val,YCFValO,_,_ = self.one_step_predict(self.valA,self.valX,self.cfValT,self.YCFVal)
        cfLossTest,cfPred_test,YCFTestO,_ ,_= self.one_step_predict(self.testA,self.testX,self.cfTestT,self.YCFTest)

        individual_effect_train, peer_effect_train, total_effect_train, \
        ate_individual_train, ate_peer_train, ate_total_train = \
            self.compute_effect_pehe(self.trainA, self.trainX, self.train_t1z1, self.train_t1z0, self.train_t0z7,
                                     self.train_t0z2, self.train_t0z0)

        individual_effect_test, peer_effect_test, total_effect_test, \
        ate_individual_test, ate_peer_test, ate_total_test = \
            self.compute_effect_pehe(self.testA, self.testX, self.test_t1z1, self.test_t1z0, self.test_t0z7,
                                     self.test_t0z2, self.test_t0z0)

        print('F_train:{:.4f}'.format(factualLossTrain.item()),
              'F_val:{:.4f}'.format(factualLossVal.item()),
              'F_test:{:.4f}'.format(factualLossTest.item()),
              'CF_train:{:.4f}'.format(cfLossTrain.item()),
              'CF_val:{:.4f}'.format(cfLossVal.item()),
              'CF_test:{:.4f}'.format(cfLossTest.item()),

              'iE_train:{:.4f}'.format(individual_effect_train.item()),
              'PE_train:{:.4f}'.format(peer_effect_train.item()),
              'TE_train:{:.4f}'.format(total_effect_train.item()),
              # 'iE_val:{:.4f}'.format(individual_effect_val.item()),
              # 'PE_val:{:.4f}'.format(peer_effect_val.item()),
              # 'TE_val:{:.4f}'.format(total_effect_val.item()),
              'iE_test:{:.4f}'.format(individual_effect_test.item()),
              'PE_test:{:.4f}'.format(peer_effect_test.item()),
              'TE_test:{:.4f}'.format(total_effect_test.item()),
              # 'dLoss_train:{:.4f}'.format(dLoss_train.item()),
              # 'dLoss_test:{:.4f}'.format(dLoss_test.item())
              )

        data = {
            "pred_train_factual":pred_train,
            "PO_train_factual":YFTrainO,
            "pred_val_factual":pred_val,
            "PO_val_factual":YFValO,
            "pred_test_factual":pred_test,
            "PO_test_factual":YFTestO,

            "pred_train_cf":cfPred_train,
            "PO_train_cf":YCFTrainO,
            "pred_val_cf":cfPred_val,
            "PO_val_cf":YCFValO,
            "pred_test_cf":cfPred_test,
            "PO_test_cf":YCFTestO,

            "factualLossTrain":factualLossTrain.detach().cpu().numpy(),
            "factualLossVal":factualLossVal.detach().cpu().numpy(),
            "factualLossTest":factualLossTest.detach().cpu().numpy(),
            "cfLossTrain":cfLossTrain.detach().cpu().numpy(),
            "cfLossVal":cfLossVal.detach().cpu().numpy(),
            "cfLossTest":cfLossTest.detach().cpu().numpy(),

            "individual_effect_train":individual_effect_train.detach().cpu().numpy(),
            "peer_effect_train":peer_effect_train.detach().cpu().numpy(),
            "total_effect_train":total_effect_train.detach().cpu().numpy(),
            # "individual_effect_val":individual_effect_val.detach().cpu().numpy(),
            # "peer_effect_val":peer_effect_val.detach().cpu().numpy(),
            # "total_effect_val":total_effect_val.detach().cpu().numpy(),
            "individual_effect_test":individual_effect_test.detach().cpu().numpy(),
            "peer_effect_test":peer_effect_test.detach().cpu().numpy(),
            "total_effect_test":total_effect_test.detach().cpu().numpy(),

            "ate_individual_train": ate_individual_train.detach().cpu().numpy(),
            "ate_peer_train": ate_peer_train.detach().cpu().numpy(),
            "ate_total_train": ate_total_train.detach().cpu().numpy(),

            "ate_individual_test": ate_individual_test.detach().cpu().numpy(),
            "ate_peer_test": ate_peer_test.detach().cpu().numpy(),
            "ate_total_test": ate_total_test.detach().cpu().numpy(),

            # "dLoss_train": dLoss_train.detach().cpu().numpy(),
            # "dLoss_test": dLoss_test.detach().cpu().numpy(),
            #
            # "save_pt": save_p_t.detach().cpu().numpy()
        }

        
        if self.args.model == "NetEsimator":
            print ("================================Save prediction...================================")
            file = "../" + self.args.dataset + "/" + self.args.model + ".pkl"

        elif self.args.model == "Mine":
            print("================================Save Mine prediction...================================")

            folder = "./RRNET/" + self.args.dataset + '/perf'


            file = folder + '/prediction_expID_' + str(self.args.expID) + '_' + '.pkl'

        elif self.args.model == "CFR_INTERFERENCE_GCN":
            folder = "../" + self.args.dataset + "/" + self.args.model

            if not os.path.exists(folder):
                os.mkdir(folder)

            file = folder + "/prediction_expID_{}.pkl".format(self.args.expID)

        else:
            print ("================================Save Bseline prediction...================================")
            file = "../" + self.args.dataset + "/" + self.args.model + "/prediction_expID_{}.pkl".format(self.args.expID)

        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file,"wb") as f:
            pkl.dump(data,f)
        print ("================================Save prediction done!================================")

    def save_curve(self):
        print ("================================Save curve...================================")
        data = {"dissTrain":self.dissTrain,
                "dissVal":self.dissVal,
                "dissTrainHalf":self.dissTrainHalf,
                "dissValHalf":self.dissValHalf,
                "diss_zTrain":self.diss_zTrain,
                "diss_zVal":self.diss_zVal,
                "diss_zTrainHalf":self.diss_zTrainHalf,
                "diss_zValHalf":self.diss_zValHalf}

        with open("../results/"+str(self.args.dataset)+"/curve/"+"curve_expID_"+str(self.args.expID)+"_alpha_"+str(self.args.alpha)+"_gamma_"+str(self.args.gamma)+"_flipRate_"+str(self.args.flipRate)+".pkl", "wb") as f:
            pkl.dump(data, f)
        print ("================================Save curve done!================================")


    def save_embedding(self):
        print ("================================Save embedding...================================")
        _, _,_, embedsTrain,_ = self.model(self.trainA, self.trainX, self.trainT)
        _, _,_, embedsTest,_ = self.model(self.testA, self.testX, self.testT)
        data = {"embedsTrain": embedsTrain.cpu().detach().numpy(), "embedsTest": embedsTest.cpu().detach().numpy(),
                "trainT": self.trainT.cpu().detach().numpy(), "testT": self.testT.cpu().detach().numpy(),
                "trainZ": self.trainZ.cpu().detach().numpy(), "testZ": self.testZ.cpu().detach().numpy()}
        with open("../results/"+str(self.args.dataset)+"/embedding/"+"embeddings_expID_"+str(self.args.expID)+"_alpha_"+str(self.args.alpha)+"_gamma_"+str(self.args.gamma)+"_flipRate_"+str(self.args.flipRate)+".pkl", "wb") as f:
            pkl.dump(data, f)
        print ("================================Save embedding done!================================")


