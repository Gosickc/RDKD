import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import datasets
import os.path as osp
from models import ImgNet, TxtNet, GCNet_IMG, GCNet_TXT, VLPTeacherNet,VLPTeacherTextNet
from utils import compress, calculate_top_map, compress_wiki, calc_map_k , p_topN_precision ,plot_precision_curve, generate_tsne, validate_model
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from DDPG import DDPG
from qlearn import LossControlEnv
from models import CriticNetwork
from global_memory import experience_pool
from seed import reset_random_seeds




class RDKD:
    def __init__(self, log, config):
        self.logger = log
        self.config = config
        self.maxfunc = torch.nn.ReLU()
        torch.manual_seed(11)
        torch.cuda.manual_seed_all(11)
        torch.cuda.set_device(self.config.GPU_ID)

        if self.config.DATASET == "MIRFlickr":
            self.train_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform)
            self.test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
            self.database_dataset = datasets.MIRFlickr(train=False, database=True, transform=datasets.mir_test_transform)

        if self.config.DATASET == "NUSWIDE":
            self.train_dataset = datasets.NUSWIDE(train=True, transform=datasets.nus_train_transform)
            self.test_dataset = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
            self.database_dataset = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)

        if self.config.DATASET == "WIKI":
            self.train_dataset = datasets.WIKI(root=self.config.DATA_DIR, train=True,
                                               transform=datasets.wiki_train_transform)
            self.test_dataset = datasets.WIKI(root=self.config.DATA_DIR, train=False,
                                              transform=datasets.wiki_test_transform)
            self.database_dataset = datasets.WIKI(root=self.config.DATA_DIR, train=True,
                                                  transform=datasets.wiki_test_transform)

        if self.config.DATASET == "MSCOCO":
            self.train_dataset = datasets.MSCOCO(train=True, transform=datasets.coco_train_transform)
            self.test_dataset = datasets.MSCOCO(train=False, database=False, transform=datasets.coco_test_transform)
            self.database_dataset = datasets.MSCOCO(train=False, database=True, transform=datasets.coco_test_transform)

        # Data Loader (Input Pipeline)
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.config.BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=self.config.NUM_WORKERS,
                                       drop_last=True)

        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.config.BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=self.config.NUM_WORKERS)

        self.database_loader = DataLoader(dataset=self.database_dataset,
                                          batch_size=self.config.BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=self.config.NUM_WORKERS)

        self.Teacher = VLPTeacherNet()
        self.ImgNet = ImgNet(bit=self.config.HASH_BIT)

        txt_feat_len = datasets.txt_feat_len

        self.TxtNet = VLPTeacherTextNet(txt_feat_len=txt_feat_len, bit=self.config.HASH_BIT)

        self.GCNet_IMG = GCNet_IMG(bit=config.HASH_BIT)
        self.GCNet_TXT = GCNet_TXT(txt_feat_len=txt_feat_len, bit=config.HASH_BIT)

        self.opt_I = torch.optim.SGD(self.ImgNet.parameters(), lr=self.config.LR_IMG, momentum=self.config.MOMENTUM,
                                      weight_decay=self.config.WEIGHT_DECAY)

        self.opt_T = torch.optim.SGD(self.TxtNet.parameters(), lr=self.config.LR_TXT, momentum=self.config.MOMENTUM,
                                      weight_decay=self.config.WEIGHT_DECAY)

        self.opt_GCN_I = torch.optim.SGD(self.GCNet_IMG.parameters(), lr=self.config.LR_IMG, momentum=self.config.MOMENTUM,
                                     weight_decay=self.config.WEIGHT_DECAY)

        self.opt_GCN_T = torch.optim.SGD(self.GCNet_TXT.parameters(), lr=self.config.LR_TXT, momentum=self.config.MOMENTUM,
                                     weight_decay=self.config.WEIGHT_DECAY)

        self.best_it = 0
        self.best_ti = 0

        self.env = LossControlEnv(config,self)  # 创建强化学习环境
        self.rl_agent = DDPG(state_dim=9, action_dim=9)  # 初始化DDPG代理
        self.state = None
        self.best_map = 0  # 所有 epoch 的最佳 mAP
        self.best_params = None  # 所有 epoch 的最佳超参数
        self.best_model_state = None  # 所有 epoch 的最佳模型状态



    def get_model_state(self, file_name='latest.pth'):
        """
        保存模型和优化器的状态到磁盘
        """
        ckp_path = osp.join(self.config.QLMODEL_DIR, file_name)
        obj = {
            'ImgNet': self.ImgNet.state_dict(),
            'TxtNet': self.TxtNet.state_dict(),
            'GCNet_IMG': self.GCNet_IMG.state_dict(),
            'GCNet_TXT': self.GCNet_TXT.state_dict(),
            'opt_I': self.opt_I.state_dict(),
            'opt_T': self.opt_T.state_dict(),
            'opt_GCN_I': self.opt_GCN_I.state_dict(),
            'opt_GCN_T': self.opt_GCN_T.state_dict(),
        }
        torch.save(obj, ckp_path)
        self.logger.info(f"********** QLSave the trained model successfully to {ckp_path}. **********")


    def get_best_model_state(self, file_name='latest.pth'):

        ckp_path = osp.join(self.config.QLMODEL_DIR, file_name)
        obj = {
            'ImgNet': self.ImgNet.state_dict(),
            'TxtNet': self.TxtNet.state_dict(),
            'GCNet_IMG': self.GCNet_IMG.state_dict(),
            'GCNet_TXT': self.GCNet_TXT.state_dict(),
            'opt_I': self.opt_I.state_dict(),
            'opt_T': self.opt_T.state_dict(),
            'opt_GCN_I': self.opt_GCN_I.state_dict(),
            'opt_GCN_T': self.opt_GCN_T.state_dict(),
        }
        torch.save(obj, ckp_path)
        self.logger.info(f"********** QLSave the best trained model successfully to {ckp_path}. **********")

    def load_model_state(self, file_name='latest.pth'):

        ckp_path = osp.join(self.config.QLMODEL_DIR, file_name)
        if not osp.exists(ckp_path):
            self.logger.warning(f"Checkpoint file {ckp_path} does not exist!")
            return

        obj = torch.load(ckp_path)
        self.ImgNet.load_state_dict(obj['ImgNet'])
        self.TxtNet.load_state_dict(obj['TxtNet'])
        self.GCNet_IMG.load_state_dict(obj['GCNet_IMG'])
        self.GCNet_TXT.load_state_dict(obj['GCNet_TXT'])
        self.opt_I.load_state_dict(obj['opt_I'])
        self.opt_T.load_state_dict(obj['opt_T'])
        self.opt_GCN_I.load_state_dict(obj['opt_GCN_I'])
        self.opt_GCN_T.load_state_dict(obj['opt_GCN_T'])
        self.logger.info(f"********** QL Loaded the model successfully from {ckp_path}. **********")

    def load_best_model_state(self, file_name='latest.pth'):

        ckp_path = osp.join(self.config.QLMODEL_DIR, file_name)
        if not osp.exists(ckp_path):
            self.logger.warning(f"Checkpoint file {ckp_path} does not exist!")
            return

        obj = torch.load(ckp_path)
        self.ImgNet.load_state_dict(obj['ImgNet'])
        self.TxtNet.load_state_dict(obj['TxtNet'])
        self.GCNet_IMG.load_state_dict(obj['GCNet_IMG'])
        self.GCNet_TXT.load_state_dict(obj['GCNet_TXT'])
        self.opt_I.load_state_dict(obj['opt_I'])
        self.opt_T.load_state_dict(obj['opt_T'])
        self.opt_GCN_I.load_state_dict(obj['opt_GCN_I'])
        self.opt_GCN_T.load_state_dict(obj['opt_GCN_T'])
        self.logger.info(f"********** QL Loaded the best model successfully from {ckp_path}. **********")


    def train_single_epoch(self, epoch):
        reset_random_seeds(seed=epoch)
        loss = 0
        for idx, (img, txt, _, index) in enumerate(self.train_loader):
            img = torch.FloatTensor(img).cuda()
            txt = torch.FloatTensor(txt.numpy()).cuda()

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            self.opt_GCN_I.zero_grad()
            self.opt_GCN_T.zero_grad()


            feat_I, hid, code_I = self.ImgNet(img)
            feat_T, code_T = self.TxtNet(txt)
            teacher_FI = self.Teacher(img)

            S, S_T_adj = self.cal_similarity_matrix(teacher_FI, txt)
            code_gcn_I = self.GCNet_IMG(teacher_FI, S)
            code_gcn_T = self.GCNet_TXT(txt, S_T_adj)


            loss = self.cal_loss(code_I, code_T, code_gcn_I, code_gcn_T, S)
            loss.backward()
            self.opt_I.step()
            self.opt_T.step()
            self.opt_GCN_I.step()
            self.opt_GCN_T.step()

            mloss = loss.item()


        current_map = validate_model(
            database_loader=self.database_loader,
            test_loader=self.test_loader,
            ImgNet=self.ImgNet,
            TxtNet=self.TxtNet,
            database_dataset=self.database_dataset,
            test_dataset=self.test_dataset,
            topk=self.config.topk

        )
        self.logger.info(f"Epoch [{epoch + 1}] - Experience pool size: {len(experience_pool)}")

        return current_map, mloss
    

    def train(self, epoch):

        self.ImgNet.set_alpha(epoch)
        self.TxtNet.set_alpha(epoch)
        self.GCNet_IMG.set_alpha(epoch)
        self.GCNet_TXT.set_alpha(epoch)

        self.ImgNet.cuda().train()
        self.TxtNet.cuda().train()
        self.GCNet_IMG.cuda().train()
        self.GCNet_TXT.cuda().train()
        self.Teacher.cuda()

        # 初始化强化学习状态
        if epoch == 0:
            self.state = self.env.reset()



        num_trials = self.config.RL_TRIALS  

        for trial in range(num_trials):
            # 保存当前模型状态以便回滚
            self.get_model_state(file_name='QL.pth')
            if trial == 0:
                # Trial == 0：不动用强化学习，只进行普通网络训练
                self.logger.info(f"Trial [{trial + 1}/{num_trials}] - Skipping reinforcement learning.")
                current_map, loss = self.train_single_epoch(epoch)  
                new_state=(
                self.config.eta,
                self.config.beta,
                self.config.lamb,
                self.config.mu,
                self.config.l1,
                self.config.l2,
                self.config.l3,
                self.config.l4,
                self.config.l5,
                )
                if current_map > self.best_map:
                    self.best_map = current_map
                    self.best_params = new_state
                    self.get_best_model_state(file_name='QLBest.pth')
                self.load_model_state(file_name='QL.pth')
                self.logger.info(
                    f"Trial [{trial + 1}/{num_trials}] - Current mAP: {current_map:.4f}, Loss: {loss:.4f}"
                )
                continue


            action = self.rl_agent.select_action(self.state,epoch)
            new_state, _, _ = self.env.step(action)


            (
                self.config.eta,
                self.config.beta,
                self.config.lamb,
                self.config.mu,
                self.config.l1,
                self.config.l2,
                self.config.l3,
                self.config.l4,
                self.config.l5,
            ) = new_state


            current_map, loss = self.train_single_epoch(epoch)


            reward = current_map - self.best_map


            if current_map > self.best_map:
                self.best_map = current_map
                self.best_params = new_state
                self.get_best_model_state(file_name='QLBest.pth')

            experience_pool.append((self.state, action, reward, new_state))
            self.state = new_state 


            self.load_model_state(file_name='QL.pth')


            if len(experience_pool) >= self.rl_agent.batch_size:
                self.logger.info(
                    f"Calling update: Experience pool size = {len(experience_pool)}, Batch size = {self.rl_agent.batch_size}")
                self.rl_agent.update(experience_pool,epoch)  # 使用全局经验池


            self.logger.info(
                f"Trial [{trial + 1}/{num_trials}] - Updated Hyperparameters: "
                f"eta={self.config.eta:.4f}, beta={self.config.beta:.4f}, lamb={self.config.lamb:.4f}, mu={self.config.mu:.4f}, "
                f"l1={self.config.l1:.4f}, l2={self.config.l2:.4f}, l3={self.config.l3:.4f}, l4={self.config.l4:.4f}, l5={self.config.l5:.4f} "
                f"Reward={reward:.4f}, Current mAP: {current_map:.4f}, Best mAP: {self.best_map:.4f}, Loss: {loss:.4f}"
            )


        if self.best_params:
            (
                self.config.eta,
                self.config.beta,
                self.config.lamb,
                self.config.mu,
                self.config.l1,
                self.config.l2,
                self.config.l3,
                self.config.l4,
                self.config.l5,
            ) = self.best_params

        self.load_best_model_state(file_name='QLBest.pth') 


        self.logger.info(f"Epoch [{epoch + 1}/{self.config.NUM_EPOCH}] - Best Hyperparameters Selected: "
                         f"eta={self.config.eta:.4f}, beta={self.config.beta:.4f}, lamb={self.config.lamb:.4f}, mu={self.config.mu:.4f}, "
                         f"l1={self.config.l1:.4f}, l2={self.config.l2:.4f}, l3={self.config.l3:.4f}, l4={self.config.l4:.4f}, l5={self.config.l5:.4f}, Best mAP: {self.best_map:.4f}")

    def eval(self):

        self.ImgNet.cuda().eval()
        self.TxtNet.cuda().eval()
        if self.config.DATASET == "WIKI":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_wiki(self.database_loader, self.test_loader,
                                                                   self.ImgNet, self.TxtNet,
                                                                   self.database_dataset, self.test_dataset)
        if self.config.DATASET == "MIRFlickr" or self.config.DATASET == "NUSWIDE" or self.config.DATASET == "MSCOCO":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.ImgNet,
                                                              self.TxtNet, self.database_dataset, self.test_dataset)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=self.config.topk)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=self.config.topk)

        if (self.best_it + self.best_ti) < (MAP_I2T + MAP_T2I):
            self.best_it = MAP_I2T
            self.best_ti = MAP_T2I

        self.logger.info('mAP@500 I->T: %.4f, mAP@500 T->I: %.4f' % (MAP_I2T, MAP_T2I))
        self.logger.info('Best MAP of I->T: %.4f, Best mAP of T->I: %.4f' % (self.best_it, self.best_ti))
        self.logger.info('--------------------------------------------------------------------')
        # 记录奖励和超参数变化


        # 计算Top-N准确率
        top_n_values = [1, 150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500]
        top_n_precisions_image_to_text = p_topN_precision(qu_BI, re_BT, qu_L, re_L, top_n_values)

        # Log and plot image-to-text top-N precisions
        self.logger.info('Image-to-Text Top-N Precisions: %s' % (top_n_precisions_image_to_text))
        # plot_precision_curve(top_n_values, top_n_precisions_image_to_text)

        # Calculate text-to-image top-N precisions
        top_n_precisions_text_to_image = p_topN_precision(qu_BT, re_BI, qu_L, re_L, top_n_values)

        # Log and plot text-to-image top-N precisions
        self.logger.info('Text-to-Image Top-N Precisions: %s' % (top_n_precisions_text_to_image))
        # plot_precision_curve(top_n_values, top_n_precisions_text_to_image)
        # Generate t-SNE Visualization
        generate_tsne(qu_BI, qu_BT)

    def cal_similarity_matrix(self, teacher_FI, txt):

        teacher_FI = F.normalize(teacher_FI, dim=1)
        S_I_Tea = teacher_FI.mm(teacher_FI.t())
        S_I = S_I_Tea * 2 - 1

        F_T = F.normalize(txt, dim=1)
        S_T = F_T.mm(F_T.t())
        S_T_adj = S_T
        S_T = S_T * 2 - 1

        S_ = self.config.eta * S_I + self.config.beta * S_T + self.config.lamb * (F.normalize(S_I).mm(F.normalize(S_T).t()))
        S = S_ * self.config.mu

        return S, S_T_adj


    # def cal_similarity_matrix(self, teacher_FI, txt):
    #     rho = 0.5  # rho 设置为 0.5
    #
    #     # 计算归一化的特征向量（余弦相似度的一部分）
    #     teacher_FI = F.normalize(teacher_FI, dim=1)
    #     F_T = F.normalize(txt, dim=1)
    #
    #     # 计算余弦相似度矩阵 C_ij
    #     S_I_Tea = teacher_FI.mm(teacher_FI.t())
    #     S_T = F_T.mm(F_T.t())
    #     # 计算欧氏距离矩阵 D_ij
    #
    #     diff_I = teacher_FI.unsqueeze(1) - teacher_FI.unsqueeze(0)
    #     dist_I = torch.sqrt(torch.sum(diff_I ** 2, dim=2))
    #     D_I = torch.exp(-torch.sqrt(dist_I) / rho)
    #
    #     diff_T = F_T.unsqueeze(1) - F_T.unsqueeze(0)
    #     dist_T = torch.sqrt(torch.sum(diff_T ** 2, dim=2))
    #     D_T = torch.exp(-torch.sqrt(dist_T) / rho)
    #
    #     # 计算最终的相似性矩阵 S_ij
    #     S_I = S_I_Tea * D_I
    #     S_T_adj = S_T
    #     S_T = S_T * D_T
    #
    #     # 按照原有的配置结合 S_I 和 S_T
    #     S_ = self.config.eta * S_I + self.config.beta * S_T + self.config.lamb * (
    #         F.normalize(S_I).mm(F.normalize(S_T).t()))
    #     S = S_ * self.config.mu
    #
    #     return S, S_T_adj

    def cal_loss(self, code_I, code_T, code_gcn_I, code_gcn_T, S):
        # Normalize codes
        B_I = F.normalize(code_I)
        B_T = F.normalize(code_T)
        B_gI = F.normalize(code_gcn_I)
        B_gT = F.normalize(code_gcn_T)

        # Calculate similarity matrices
        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        BI_BT = B_I.mm(B_T.t())

        GBI_GBI = B_gI.mm(B_gI.t())
        GBT_GBT = B_gT.mm(B_gT.t())
        GBI_GBT = B_gI.mm(B_gT.t())



        # Calculate different losses
        Hashing_level_loss = F.mse_loss(B_I, B_gI) + F.mse_loss(B_T, B_gT)
        Intra_modal_loss = F.mse_loss(BI_BI, S) + F.mse_loss(BT_BT, S)
        Cross_modal_loss = F.mse_loss(BI_BT, S) + F.mse_loss(GBI_GBT, BI_BT) - (B_I * B_T).sum(dim=1).mean() - (
                B_gI * B_gT).sum(dim=1).mean()
        Graph_level_loss = F.mse_loss(S, GBI_GBI) + F.mse_loss(S, GBT_GBT)

        KD_loss = self.config.l1 * Hashing_level_loss + self.config.l2 * Intra_modal_loss + self.config.l3 * Cross_modal_loss + self.config.l4 * Graph_level_loss

        # Calculate thresholds
        k = torch.tensor(self.config.HASH_BIT, dtype=torch.float32)
        thresh = (1 - S) * k / 2
        width = 3
        up_thresh = thresh
        low_thresh = thresh - width
        low_thresh[low_thresh <= 0] = 0
        low_thresh[S == -1] = self.config.HASH_BIT / 2

        # Initialize flag matrices
        low_flag = torch.ones(self.config.BATCH_SIZE, self.config.BATCH_SIZE).cuda()
        up_flag = torch.ones(self.config.BATCH_SIZE, self.config.BATCH_SIZE).cuda()
        low_flag[S == 1] = 0
        low_flag[S == -1] = self.config.BETA
        up_flag[S == -1] = 0
        up_flag[S == 1] = self.config.ALPHA

        GBI_1GBI = (self.config.HASH_BIT - B_gI.mm(B_gI.t())) / 2
        GBT_1GBT = (self.config.HASH_BIT - B_gT.mm(B_gT.t())) / 2
        GBI_1GBT = (self.config.HASH_BIT - B_gI.mm(B_gT.t())) / 2
        GBT_1GBI = (self.config.HASH_BIT - B_gT.mm(B_gI.t())) / 2
        BI_1BI = (self.config.HASH_BIT - B_I.mm(B_I.t())) / 2
        BT_1BT = (self.config.HASH_BIT - B_T.mm(B_T.t())) / 2
        BI_1BT = (self.config.HASH_BIT - B_I.mm(B_T.t())) / 2
        BT_1BI = (self.config.HASH_BIT - B_T.mm(B_I.t())) / 2
        # Calculate lower bound loss
        loss1 = (torch.norm(self.maxfunc(low_thresh - BI_1BI) * low_flag) +
                 torch.norm(self.maxfunc(low_thresh - BT_1BT) * low_flag) +
                 torch.norm(self.maxfunc(low_thresh - BT_1BI) * low_flag) +
                 torch.norm(self.maxfunc(low_thresh - BI_1BT) * low_flag)) / (
                            self.config.BATCH_SIZE * self.config.BATCH_SIZE)

        # Calculate upper bound loss
        loss2 = (torch.norm(self.maxfunc(BI_1BI - up_thresh) * up_flag) +
                 torch.norm(self.maxfunc(BT_1BT - up_thresh) * up_flag) +
                 torch.norm(self.maxfunc(BT_1BI - up_thresh) * up_flag) +
                 torch.norm(self.maxfunc(BI_1BT - up_thresh) * up_flag)) / (
                            self.config.BATCH_SIZE * self.config.BATCH_SIZE)
        loss3 = (torch.norm(self.maxfunc(low_thresh - GBI_1GBI) * low_flag) +
                 torch.norm(self.maxfunc(low_thresh - GBT_1GBT) * low_flag) +
                 torch.norm(self.maxfunc(low_thresh - GBT_1GBI) * low_flag) +
                 torch.norm(self.maxfunc(low_thresh - GBI_1GBT) * low_flag)) / (
                            self.config.BATCH_SIZE * self.config.BATCH_SIZE)

        # Calculate upper bound loss
        loss4 = (torch.norm(self.maxfunc(GBI_1GBI - up_thresh) * up_flag) +
                 torch.norm(self.maxfunc(GBT_1GBT - up_thresh) * up_flag) +
                 torch.norm(self.maxfunc(GBT_1GBI - up_thresh) * up_flag) +
                 torch.norm(self.maxfunc(GBI_1GBT - up_thresh) * up_flag)) / (
                            self.config.BATCH_SIZE * self.config.BATCH_SIZE)

        wdloss=loss1+loss2
        # Combine losses
        total_loss = KD_loss + self.config.l5 * wdloss #+ loss3 + loss4

        return total_loss



    def save_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.ImgNet.state_dict(),
            'TxtNet': self.TxtNet.state_dict(),
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')


    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
            self.ImgNet.load_state_dict(obj['ImgNet'])
            self.TxtNet.load_state_dict(obj['TxtNet'])
        except IOError:
            self.logger.error('********** Fail to load checkpoint %s!*********' % ckp_path)
            raise IOError


