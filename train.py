import os
import torch
# from rdkit.Chem.BRICS import ps
import torch.nn.functional as F
from network import Network
from metric import valid, inference
from torch.utils.data import Dataset
import numpy as np
import argparse
from loss import Loss,  Proto_Align_Loss1, hard_sample_aware_infoNCE, compute_sdm
from dataloader import load_data
import os
import time
import random
from kmeans_gpu import kmeans
from sklearn.cluster import MiniBatchKMeans
import warnings
import itertools

warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans




os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# Dataname = 'KUST_ET'
# Dataname = 'KUST_CT'
# Dataname = 'KUST_CV'
# Dataname = 'reuters'
Dataname = 'KUST_CE'
# Dataname = 'KUST_CB'
# Dataname = 'KUST_BT'
# Dataname = 'KUST_CL'

CL_Loss = ['InfoNCE', 'PSCL', 'RINCE']  # three kinds of contrastive losses
Measure_M_N = ['CMI', 'JSD',
               'MMD']  # Class Mutual Information (CMI), Jensen–Shannon Divergence (JSD), Maximum Mean Discrepancy (MMD)
sample_mmd = 2000  # select partial samples to compute MMD as it has high complexity, otherwise might be out-of-memory
Reconstruction = ['AE', 'DAE', 'MAE']  # autoencoder (AE), denoising autoencoder (DAE), masked autoencoder (MAE)
per = 0.3  # the ratio of masked samples to perform masked AE, e.g., 30%

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)

parser.add_argument('--batch_size', default=256, type=int)  # 256
parser.add_argument("--temperature_f", default=1.0)  # 1.0
parser.add_argument("--contrastive_loss", default=CL_Loss[0])  # 0, 1, 2
parser.add_argument("--measurement", default=Measure_M_N[0])  # 0, 1, 2
parser.add_argument("--Recon", default=Reconstruction[0])  # 0, 1, 2
parser.add_argument("--bi_level_iteration", default=4)  # 2
parser.add_argument("--times_for_K", default=1)  # 0.5 1 2 4
parser.add_argument("--Lambda", default=1)  # 0.001 0.01 0.1 1 10 100 1000
parser.add_argument("--gama", default=1)  # 0.001 0.01 0.1 1 10 100 1000
parser.add_argument("--alpha", default=0.001)
parser.add_argument("--beta1", default=0.01)
parser.add_argument("--learning_rate", default=0.0003)  # 0.0003
parser.add_argument("--weight_decay", default=0.)  # 0.
parser.add_argument("--workers", default=8)  # 8
parser.add_argument("--mse_epochs", default=100)  # 100
parser.add_argument("--con_epochs", default=50)  # 100
parser.add_argument("--feature_dim", default=512)  # 512
parser.add_argument("--high_feature_dim", default=128)  # 128
parser.add_argument('--tao', type=float, default=0.9, help='high confidence rate')
parser.add_argument('--beta', type=float, default=1, help='focusing factor beta')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





if args.dataset == "KUST_CE":   # KUST-CE  α：0.001 β：0.01
    args.mse_epochs = 100
    args.con_epochs = 100
    args.bi_level_iteration = 1
    seed = 42
if args.dataset == "KUST_CL":   #KUST-CL α：0.001 β：1
    args.mse_epochs = 100
    args.con_epochs = 100
    args.bi_level_iteration = 3
    args.alpha = 0.001
    args.beta1 = 1
    seed = 42
if args.dataset == "KUST_CB":  #KUST-CB α：0.001 β：1
    args.mse_epochs = 100
    args.con_epochs = 100
    args.bi_level_iteration = 2
    args.alpha = 0.001
    args.beta1 = 1
    seed = 42
if args.dataset == "KUST_CV":  #KUST-CV α：0.001 β：0.01
    args.mse_epochs = 100
    args.con_epochs = 100
    args.bi_level_iteration = 1
    args.alpha = 0.001
    args.beta1 = 0.01
    seed = 42
if args.dataset == "KUST_CT":  #KUST-CT α：0.001 β：0.01
    args.mse_epochs = 100
    args.con_epochs = 100
    args.bi_level_iteration = 2
    seed = 42
if args.dataset == "KUST_BT":  #KUST-BT α：0.001 β：0.01
    args.mse_epochs = 100
    args.con_epochs = 100
    args.bi_level_iteration = 2
    seed = 42
if args.dataset == "KUST_ET":  #KUST-ET α：0.001 β：0.01
    args.mse_epochs = 100
    args.con_epochs = 100
    args.bi_level_iteration = 1
    seed = 42
if args.dataset == "reuters":
    args.mse_epochs = 200  # 200
    args.con_epochs = 50  # 50
    args.bi_level_iteration = 1
    seed = 42

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed)

Total_con_epochs = args.con_epochs * args.bi_level_iteration


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def square_euclid_distance(Z, center):
    # 确保 Z 和 center 都在同一个设备上
    Z = Z.to('cuda')
    center = center.to('cuda')

    ZZ = (Z * Z).sum(-1).reshape(-1, 1).repeat(1, center.shape[0])
    CC = (center * center).sum(-1).reshape(1, -1).repeat(Z.shape[0], 1)
    ZZ_CC = ZZ + CC
    ZC = Z @ center.T
    distance = ZZ_CC - 2 * ZC
    return distance



def high_confidence(Z, center):
    # 确保 Z 和 center 都在同一个设备上
    Z = Z.to('cuda')
    center = center.to('cuda')

    distance_norm = torch.min(F.softmax(square_euclid_distance(Z, center), dim=1),
                              dim=1).values
    value, _ = torch.topk(distance_norm, int(
        Z.shape[0] * (1 - args.tao)))
    index = torch.where(distance_norm <= value[-1],
                        torch.ones_like(distance_norm), torch.zeros_like(distance_norm))

    high_conf_index_v1 = torch.nonzero(index).reshape(-1, )
    high_conf_index_v2 = high_conf_index_v1 + Z.shape[0]
    H = torch.cat([high_conf_index_v1, high_conf_index_v2], dim=0)
    H_mat = np.ix_(H.cpu(), H.cpu())

    return H, H_mat


def pseudo_matrix(P, S, node_num):
    # P = torch.tensor(P)
    P = P.clone().detach()
    P = torch.cat([P, P], dim=0)
    Q = (P == P.unsqueeze(1)).float().to(device)
    S_norm = (S - S.min()) / (S.max() - S.min())
    M_mat = torch.abs(Q - S_norm) ** args.beta
    M = torch.cat([torch.diag(M_mat, node_num), torch.diag(M_mat, -node_num)], dim=0)
    return M, M_mat


accs = []
nmis = []
aris = []
purs = []
ACC_tmp = 0
losses = []
for Runs in range(1):  # 10
    print("ROUND:{}".format(Runs + 1))

    t1 = time.time()
    # setup_seed(5)   # if we find that the initialization of networks is sensitive, we can set a seed for stable performance.
    dataset, dims, view, data_size, class_num = load_data(args.dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # drop_last=True,
        drop_last=False,
    )


    def comprehensive_similarity(Z1, Z2):
        Z1_Z2 = torch.cat([torch.cat([Z1 @ Z1.T, Z1 @ Z2.T], dim=1),
                           torch.cat([Z2 @ Z1.T, Z2 @ Z2.T], dim=1)], dim=0)

        S = Z1_Z2
        return S


    def Low_level_rec_train(epoch, rec='AE', p=0.3, mask_ones_full=[], mask_ones_not_full=[]):
        tot_loss = 0.
        criterion = torch.nn.MSELoss()
        Vones_full = []
        Vones_not_full = []
        flag_full = 0
        flag_not_full = 0
        for batch_idx, (xs, _, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)

            xnum = xs[0].shape[0]

            if rec == 'AE':
                optimizer.zero_grad()
                _, _, xrs, _, _ = model(xs)

            loss_list = []
            for v in range(view):
                loss_list.append(criterion(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
        return Vones_full, Vones_not_full


    def High_level_contrastive_train(epoch, nmi_matrix, Lambda=1.0, gama=1.0, alpha=0.001, beta1=0.01, rec='AE', p=0.3,
                                     mask_ones_full=[],
                                     mask_ones_not_full=[]):
        tot_loss = 0.
        mes = torch.nn.MSELoss()
        record_loss_con = []
        Vones_full = []
        Vones_not_full = []
        flag_full = 0
        flag_not_full = 0

        for v in range(view):
            record_loss_con.append([])
            for w in range(view):
                record_loss_con[v].append([])

        for batch_idx, (xs, _, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)

            optimizer.zero_grad()
            zs, qs, xrs, hs, re_h = model(xs)
            loss_list = []
            xnum = xs[0].shape[0]
            for v in range(view):
                for w in range(view):
                    # ###############prototype
                    # '''修改11'''
                    ps = model.forward_prototype(xs)
                    '''prototype修改'''
                    criterion_proto = Proto_Align_Loss1().to(device)
                    l_tmp = criterion_proto(ps[v], ps[w])
                    # loss_list.append(0.001* l_tmp)
                    alpha = args.alpha
                    loss_list.append(alpha * l_tmp)

                    ###############prototype 变体消融
                    # '''修改11'''
                    # ps = model.forward_prototype(xs)
                    # loss_list.append(kl_divergence(ps[v], ps[w]))

                    ###############hard sample
                    '''hard sample22'''
                    pos_weight = torch.ones(zs[v].shape[0] * 2).to(device)
                    pos_neg_weight = torch.ones([zs[v].shape[0] * 2, zs[v].shape[0] * 2]).to(device)
                    S = comprehensive_similarity(zs[v], zs[w]).to(device)
                    mask = torch.ones([zs[v].shape[0] * 2, zs[v].shape[0] * 2]).to(device) - torch.eye(
                        zs[v].shape[0] * 2).to(device)
                    l_tmp = hard_sample_aware_infoNCE(S, mask, pos_neg_weight, pos_weight, zs[v].shape[0])
                    loss_list.append(l_tmp)

                    loss_list.append(mes(xs[v], xrs[v]))


                logit_scale = 0.1
                Z = (zs[v] + zs[w]) / 2
                P, center = kmeans(Z, class_num, distance="euclidean")
                P = P.to(device)
                l_tmp = compute_sdm(zs[v], zs[w], P, logit_scale)

                beta1 = args.beta1
                loss_list.append(beta1 * l_tmp)


            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

        for v in range(view):
            for w in range(view):
                Z = (zs[v] + zs[w]) / 2
                P, center = kmeans(Z, class_num, distance="euclidean")

                H, H_mat = high_confidence(Z, center)
                M, M_mat = pseudo_matrix(P, S, zs[v].shape[0])
                M = M.to(device)
                M_mat = M_mat.to(device)
                H = H.to(device)
                pos_weight[H] = M[H].data.to(device)
                # pos_weight[H]= pos_weight[H].to(device)

                pos_neg_weight[H_mat] = M_mat[H_mat].data.to(device)
                # pos_neg_weight[H_mat]=pos_neg_weight[H_mat].to(device)



        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
        return Vones_full, Vones_not_full, record_loss_con, _


    if not os.path.exists('./models'):
        os.makedirs('./models')

    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)

    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, device).to(device)

    print("Initialization......")
    epoch = 0
    while epoch < args.mse_epochs:
        Low_level_rec_train(epoch)
        loss = Low_level_rec_train(epoch)
        losses.append(loss)
        epoch += 1
        if epoch == 1:
            mask_ones_full, mask_ones_not_full = Low_level_rec_train(epoch,
                                                                     rec=args.Recon,
                                                                     p=per,
                                                                     )
        else:
            Low_level_rec_train(epoch,
                                rec=args.Recon,
                                p=per,
                                mask_ones_full=mask_ones_full,
                                mask_ones_not_full=mask_ones_not_full,
                                )

    acc, nmi, ari, pur, nmi_matrix_1, _ = valid(model, device, dataset, view, data_size, class_num,
                                                eval_h=True, eval_z=False, times_for_K=args.times_for_K,
                                                Measure=args.measurement, test=False, sample_num=sample_mmd)

    Iteration = 1
    print("Iteration " + str(Iteration) + ":")
    epoch = 0
    record_loss_con = []
    record_cos = []
    while epoch < Total_con_epochs:
        # High_level_contrastive_train(epoch, Lambda=1.0)
        # loss = High_level_contrastive_train(epoch)
        # losses.append(loss)
        epoch += 1
        if epoch == 1:
            mask_ones_full, mask_ones_not_full, record_loss_con_, record_cos_ = High_level_contrastive_train(epoch,
                                                                                                             nmi_matrix_1,
                                                                                                             args.Lambda,
                                                                                                             rec=args.Recon,
                                                                                                             p=per)
        else:
            _, _, record_loss_con_, record_cos_ = High_level_contrastive_train(epoch,
                                                                               nmi_matrix_1,
                                                                               args.Lambda,
                                                                               alpha=args.alpha,
                                                                               beta1=args.beta1,
                                                                               rec=args.Recon,
                                                                               p=per,
                                                                               mask_ones_full=mask_ones_full,
                                                                               mask_ones_not_full=mask_ones_not_full,
                                                                               )
        # print(f"alpha: {args.alpha}, beta1: {args.beta1}")

        record_loss_con.append(record_loss_con_)
        record_cos.append(record_cos_)
        if epoch % args.con_epochs == 0:

            if epoch == args.mse_epochs + Total_con_epochs:
                break

            # print(nmi_matrix_1)
            ###这里每个epoch都输出前移了1行
            acc, nmi, ari, pur, _, nmi_matrix_2 = valid(model, device, dataset, view, data_size, class_num,
                                                        eval_h=False, eval_z=True, times_for_K=args.times_for_K,
                                                        Measure=args.measurement, test=False, sample_num=sample_mmd)
            nmi_matrix_1 = nmi_matrix_2
            if epoch < Total_con_epochs:
                Iteration += 1
                print("Iteration " + str(Iteration) + ":")

        pg = [p for p in model.parameters() if p.requires_grad]
        #  this code matters, to re-initialize the optimizers
        optimizer = torch.optim.Adam(pg, lr=args.learning_rate, weight_decay=args.weight_decay)
    ######
    accs.append(acc)
    nmis.append(nmi)
    aris.append(ari)
    purs.append(pur)

    # if acc > ACC_tmp:
    #     ACC_tmp = acc
    #     state = model.state_dict()
    #     torch.save(state, './models/' + args.dataset + '.pth')

    t2 = time.time()
    print("Time cost: " + str(t2 - t1))
    print('End......')

