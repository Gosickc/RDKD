import torch
import math
import torch.nn as nn
import torchvision
import clip
from pygcn.layers import GraphConvolution
import torch.nn.functional as F

class VLPTeacherNet(nn.Module):
    def __init__(self):
        super(VLPTeacherNet, self).__init__()
        self.clip_image_encode, _ = clip.load("ViT-B/16", device='cuda:0')

    def forward(self, x):
        with torch.no_grad():
            feat_I = self.clip_image_encode.encode_image(x)
            feat_I = feat_I.type(torch.float32)
        return feat_I


class VLPTeacherTextNet(nn.Module):
    def __init__(self, txt_feat_len, bit):
        super(VLPTeacherTextNet, self).__init__()

        # 定义多层线性层
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, bit)
        self.alpha = 1.0

    def forward(self, text):
        text = text.type(torch.float32)  # 确保类型为 float32

        # 通过多层感知机 (MLP)
        mid_feat_T = torch.relu(self.fc1(text))  # 第一层全连接，ReLU激活
        hid = torch.relu(self.fc2(mid_feat_T))  # 第二层全连接，ReLU激活
        hid = self.fc3(hid)  # 第三层全连接
        code = torch.tanh(self.alpha * hid)  # 通过tanh得到bit哈希码

        return text, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class ImgNet(nn.Module):
    def __init__(self, bit):
        super(ImgNet, self).__init__()
        self.VGG = torchvision.models.vgg16(pretrained=True)
        self.VGG.classifier = self.VGG.classifier[:-1]
        self.feature_layer = nn.Linear(4096, 4096)
        self.hash_layer = nn.Linear(4096, bit)
        self.alpha = 1.0

    def forward(self, x):
        feat_I = self.VGG(x)
        hid = self.feature_layer(feat_I)
        mid = self.hash_layer(torch.relu(hid))
        code = torch.tanh(self.alpha * mid)
        return feat_I, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class TxtNet(nn.Module):
    def __init__(self, txt_feat_len, bit):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, bit)
        self.alpha = 1.0

    def forward(self, x):
        mid_feat_T = torch.relu(self.fc1(x))
        hid = torch.relu(self.fc2(mid_feat_T))
        hid = self.fc3(hid)
        code = torch.tanh(self.alpha * hid)
        return x, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GCNet_IMG(nn.Module):
    def __init__(self, bit):
        super(GCNet_IMG, self).__init__()

        self.gc1 = GraphConvolution(512, 4096)
        self.gc2 = GraphConvolution(4096, bit)
        self.alpha = 1.0

    def forward(self, x, adj):
        x = torch.relu(self.gc1(x, adj))
        feat_G_I = self.gc2(x, adj)
        code = torch.tanh(self.alpha * feat_G_I)
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GCNet_TXT(nn.Module):
    def __init__(self, txt_feat_len, bit):
        super(GCNet_TXT, self).__init__()

        self.gc1 = GraphConvolution(txt_feat_len, 4096)
        self.gc2 = GraphConvolution(4096, bit)
        self.alpha = 1.0

    def forward(self, x, adj):
        x = torch.relu(self.gc1(x, adj))
        feat_G_T = self.gc2(x, adj)
        code = torch.tanh(self.alpha * feat_G_T)
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return F.relu(x + residual)

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_residual_blocks=3):
        super(ActorNetwork, self).__init__()
        self.fc_input = nn.Linear(state_dim, hidden_dim)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_residual_blocks)]
        )
        self.fc_output = nn.Linear(hidden_dim, action_dim)
        self.activation = nn.Tanh()

    def forward(self, state, epoch):
        """
        动态限制 action 的输出范围。
        随着 epoch 增加，scaling_factor 逐渐减小。
        """
        x = F.relu(self.fc_input(state))
        for block in self.residual_blocks:
            x = block(x)

        # 动态缩放因子，随着 epoch 增加逐渐减小
        scaling_factor = max(0.1 / (1 + epoch * 0.01), 0.01)  # 最小缩放为 0.01

        # 限制 action 范围到 [-scaling_factor, scaling_factor]
        action = self.activation(self.fc_output(x)) * scaling_factor
        return action


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_residual_blocks=3):
        super(CriticNetwork, self).__init__()
        self.fc_input = nn.Linear(state_dim + action_dim, hidden_dim)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_residual_blocks)]
        )
        self.fc_output = nn.Linear(hidden_dim, 1)  # 输出 Q 值

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # 将状态和动作拼接
        x = F.relu(self.fc_input(x))
        for block in self.residual_blocks:
            x = block(x)
        value = self.fc_output(x)
        return value


