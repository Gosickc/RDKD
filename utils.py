import torch
from torch.autograd import Variable
import numpy as np
import logging
import os.path as osp
from args import config
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def calc_hamming_dist(B1, B2):
    """
       :param B1:  vector [n]
       :param B2:  vector [r*n]
       :return: hamming distance [r]
       """
    leng = B2.shape[1]  # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.T))
    return distH

def p_topK(qB, rB, query_label, retrieval_label, K=None):
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    num_query = query_label.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm)[1][:total]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p

def compress_wiki(train_loader, test_loader, modeli, modelt, train_dataset, test_dataset, classes=10):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, target, _) in enumerate(train_loader):
        var_data_I = Variable(data_I.cuda())
        _,_,code_I = modeli(var_data_I)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        re_L.extend(target)
        
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _,_,code_T = modelt(var_data_T)
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, target, _) in enumerate(test_loader):
        var_data_I = Variable(data_I.cuda())
        _,_,code_I = modeli(var_data_I)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        qu_L.extend(target)
        
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _,_,code_T = modelt(var_data_T)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = np.eye(classes)[np.array(re_L)]

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = np.eye(classes)[np.array(qu_L)]
    
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def compress(database_loader, test_loader, model_I, model_T, database_dataset, test_dataset):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, _, _) in enumerate(database_loader):
        var_data_I = Variable(data_I.cuda())
        _, _, code_I = model_I(var_data_I)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _, code_T = model_T(var_data_T)
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())


    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, _, _) in enumerate(test_loader):
        var_data_I = Variable(data_I.cuda())
        _, _, code_I = model_I(var_data_I)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _, code_T = model_T(var_data_T)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())


    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = database_dataset.train_labels

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = test_dataset.train_labels
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L




def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calc_hamming_dist(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    query_L = torch.from_numpy(query_L)
    retrieval_L = torch.from_numpy(retrieval_L)
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)      # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB)
        hamm = torch.from_numpy(hamm)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj




def logger():
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    name = config.CHECKPOINT.split('.')[0]
    log_name = name + '@' + str(config.topk) + '.log'
    log_dir = './logs'
    txt_log = logging.FileHandler(osp.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    return logger


def p_topN_precision(qB, rB, query_label, retrieval_label, K=None):
    qB = np.array(qB)
    rB = np.array(rB)
    query_label = np.array(query_label)
    retrieval_label = np.array(retrieval_label)
    num_query = query_label.shape[0]
    precisions = [0] * len(K)

    for iter in range(num_query):
        q_L = torch.from_numpy(query_label[iter].reshape(1, -1))
        r_L = torch.from_numpy(retrieval_label)
        gnd = (q_L.mm(r_L.t()) > 0).float().squeeze().numpy()
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            if K[i] == 0:
                precisions[i] = np.nan  # 为 K=0 设置为 NaN 或 0
                continue  # 忽略 K 为 0 的情况
            total = min(K[i], retrieval_label.shape[0])
            if total == 0:
                precisions[i] = np.nan  # 为 total=0 设置为 NaN 或 0
                continue  # 忽略 total 为 0 的情况
            ind = np.argsort(hamm)[:total]
            gnd_ = gnd[ind]
            precisions[i] += np.sum(gnd_) / total

    precisions = np.array(precisions) / num_query
    return precisions


# 绘制 Top-N Precision 曲线
def plot_precision_curve(top_n_values, precisions):
    plt.figure(figsize=(10, 6))
    plt.plot(top_n_values, precisions, marker='o')
    plt.title('Top-N Precision Curve')
    plt.xlabel('Number of Retrieved Samples')
    plt.ylabel('Precision')
    plt.xticks(np.arange(0, max(top_n_values) + 1, step=500))
    plt.grid(True)
    plt.show()


def generate_tsne(image_features, text_features):
    """
    Generate and plot t-SNE visualizations for image and text features.
    """

    # Stack image and text features for t-SNE
    combined_features = np.vstack((image_features, text_features))

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(combined_features)

    # Split results back to image and text parts
    tsne_image = tsne_results[:len(image_features)]
    tsne_text = tsne_results[len(image_features):]

    # Plotting
    plt.figure(figsize=(10, 8))

    # Image Features: Use a lighter blue
    plt.scatter(tsne_image[:, 0], tsne_image[:, 1], c='#4682b4', marker='o', label='Image', alpha=0.6, s=20)

    # Text Features: Use a lighter yellow
    plt.scatter(tsne_text[:, 0], tsne_text[:, 1], c='#ffd700', marker='o', label='Text', alpha=0.6, s=20)

    plt.legend()
    plt.title("t-SNE Visualization of Image and Text Features")

    plt.savefig("tsne_visualization_mir.svg", format='svg')
    plt.show()


def validate_model(database_loader, test_loader, ImgNet, TxtNet, database_dataset, test_dataset, topk):
        """
        验证模型性能，返回用于强化学习的奖励值。

        :param database_loader: 数据库集的 DataLoader
        :param test_loader: 测试集的 DataLoader
        :param ImgNet: 图像哈希模型
        :param TxtNet: 文本哈希模型
        :param database_dataset: 数据库数据集
        :param test_dataset: 测试数据集
        :param topk: 用于计算 mAP 的 top-k 值
        :return: reward (平均 mAP 值)
        """
        # 切换模型到评估模式
        ImgNet.eval()
        TxtNet.eval()

        # 使用函数 compress 获取图像和文本的哈希码以及标签
        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(database_loader, test_loader, ImgNet, TxtNet,
                                                          database_dataset, test_dataset)

        # 计算图像到文本（I->T）的 mAP
        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=topk)

        # 计算文本到图像（T->I）的 mAP
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=topk)

        # 平均 mAP 作为奖励
        avg_map = (MAP_I2T + MAP_T2I) / 2

        print(f"Validation Results: MAP_I2T: {MAP_I2T:.4f}, MAP_T2I: {MAP_T2I:.4f}, avg_map: {avg_map:.4f}")

        return avg_map



def get_model_state(self):
    return {
        'ImgNet': self.ImgNet.state_dict(),
        'TxtNet': self.TxtNet.state_dict(),
        'GCNet_IMG': self.GCNet_IMG.state_dict(),
        'GCNet_TXT': self.GCNet_TXT.state_dict(),
        'opt_I': self.opt_I.state_dict(),
        'opt_T': self.opt_T.state_dict(),
        'opt_GCN_I': self.opt_GCN_I.state_dict(),
        'opt_GCN_T': self.opt_GCN_T.state_dict(),
    }
def load_model_state(self, state):
    self.ImgNet.load_state_dict(state['ImgNet'])
    self.TxtNet.load_state_dict(state['TxtNet'])
    self.GCNet_IMG.load_state_dict(state['GCNet_IMG'])
    self.GCNet_TXT.load_state_dict(state['GCNet_TXT'])
    self.opt_I.load_state_dict(state['opt_I'])
    self.opt_T.load_state_dict(state['opt_T'])
    self.opt_GCN_I.load_state_dict(state['opt_GCN_I'])
    self.opt_GCN_T.load_state_dict(state['opt_GCN_T'])
