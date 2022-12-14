import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
class encoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(encoder, self).__init__()
        # print(n_dim,dims[0])
        self.enc_1 = Linear(n_dim, dims[0])
        self.enc_2 = Linear(dims[0], dims[1])
        self.enc_3 = Linear(dims[1], dims[2])
        self.z_layer = Linear(dims[2], n_z)
        self.z_b0 = nn.BatchNorm1d(n_z)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_b0(self.z_layer(enc_h3))
        return z


class decoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(decoder, self).__init__()
        self.dec_0 = Linear(n_z, n_z)
        self.dec_1 = Linear(n_z, dims[2])
        self.dec_2 = Linear(dims[2], dims[1])
        self.dec_3 = Linear(dims[1], dims[0])
        self.x_bar_layer = Linear(dims[0], n_dim)

    def forward(self, z):
        r = F.relu(self.dec_0(z))
        dec_h1 = F.relu(self.dec_1(r))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar


class AE(nn.Module):

    def __init__(self, n_stacks, n_input, n_z, nLabel):
        super(AE, self).__init__()
        # dims0 = []
        # for idim in range(n_stacks-2):
        #     linshidim=round(n_input[0]*0.8)
        #     linshidim = int(linshidim)
        #     dims0.append(linshidim)
        # linshidim = 1500
        # linshidim = int(linshidim)
        # dims0.append(linshidim)

        dims = []
        for n_dim in n_input:

            linshidims = []
            for idim in range(n_stacks - 2):
                linshidim = round(n_dim * 0.8)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1500)
            dims.append(linshidims)

        self.encoder_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.decoder_list = nn.ModuleList([decoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        # encoder0
        # self.enc0_1 = Linear(n_input[0], dims0[0])
        # self.enc0_2 = Linear(dims0[0], dims0[1])
        # self.enc0_3 = Linear(dims0[1], dims0[2])
        # self.z0_layer = Linear(dims0[2], n_z)
        # self.z0_b0 = nn.BatchNorm1d(n_z)

        # # decoder0
        # self.dec0_0 = Linear(n_z, n_z)
        # self.dec0_1 = Linear(n_z, dims0[2])
        # self.dec0_2 = Linear(dims0[2], dims0[1])
        # self.dec0_3 = Linear(dims0[1], dims0[0])
        # self.x0_bar_layer = Linear(dims0[0], n_input[0])

        self.regression = Linear(n_z, nLabel)
        self.act = nn.Sigmoid()

    def forward(self, mul_X, we):

        # enc0_h1 = F.relu(self.enc0_1(x0))
        # enc0_h2 = F.relu(self.enc0_2(enc0_h1))
        # enc0_h3 = F.relu(self.enc0_3(enc0_h2))
        # z0 = self.z0_b0(self.z0_layer(enc0_h3))

        summ = 0
        individual_zs = []
        for enc_i, enc in enumerate(self.encoder_list):
            z_i = enc(mul_X[enc_i])
            individual_zs.append(z_i)
            summ += torch.diag(we[:, enc_i]).mm(z_i)
        # summ = torch.diag(we[:,0]).mm(z0)+torch.diag(we[:,1]).mm(z1)+torch.diag(we[:,2]).mm(z2)+torch.diag(we[:,3]).mm(z3)\
        # +torch.diag(we[:,4]).mm(z4)+torch.diag(we[:,5]).mm(z5)
        wei = 1 / torch.sum(we, 1)
        z = torch.diag(wei).mm(summ)

        # # decoder0
        # r0 = F.relu(self.dec0_0(z))
        # dec0_h1 = F.relu(self.dec0_1(r0))
        # dec0_h2 = F.relu(self.dec0_2(dec0_h1))
        # dec0_h3 = F.relu(self.dec0_3(dec0_h2))
        # x0_bar = self.x0_bar_layer(dec0_h3)

        x_bar_list = []
        for dec_i, dec in enumerate(self.decoder_list):
            x_bar_list.append(dec(individual_zs[dec_i]))
            # x_bar_list.append(dec(z))
        yLable = self.act(self.regression(F.relu(z)))
        return x_bar_list, yLable, z, individual_zs


class DICNet(nn.Module):

    def __init__(self,
                 n_stacks,
                 n_input,
                 n_z,
                 Nlabel):
        super(DICNet, self).__init__()

        self.ae = AE(
            n_stacks=n_stacks,
            n_input=n_input,
            n_z=n_z,
            nLabel=Nlabel)

    def forward(self, mul_X, we):
        x_bar_list, target_pre, fusion_z, individual_zs = self.ae(mul_X, we)

        return x_bar_list, target_pre, fusion_z, individual_zs