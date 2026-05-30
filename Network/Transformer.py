import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import time


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # t0 = time.time()

            x = attn(x) + x
            x = ff(x) + x
            # t1 = time.time()
            # print(f"[TransInner Timer] 总用时: {t1 - t0:.8f} 秒")


        return self.norm(x)

class Branch(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, token, feature_length, dim_head, dropout):
        super().__init__()
        self.conv_net = nn.Conv1d(1, token, kernel_size=token, stride=token)
        self.transformer = Transformer(feature_length, depth, heads, dim_head, mlp_dim, dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, token + 1, dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
  
    def forward(self, x):
        b, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(x.shape[1])]
        x = self.transformer(x)
        # x = self.leaky_relu(x)
        return x

class MyMultiBranchModel(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dim_head=16, dropout=0.):
        super().__init__()
        # 定义四个不同配置的卷积网络和Transformer
        self.branches = nn.ModuleList([
            #dim, depth, heads, mlp_dim, token, feature_length, dim_head, dropout
            Branch(64, depth, 16, mlp_dim, 64, 64, 64//16, dropout),
            Branch(128, depth, 32, mlp_dim, 32, 128, 128//32, dropout),
            Branch(256, depth, 64, mlp_dim, 16, 256, 256//64, dropout),
            Branch(512, depth, 128, mlp_dim, 8, 512, 512//128, dropout),
        ])
        # 定义卷积层
        self.conv_layer1 = nn.Conv1d(
            in_channels=4,      # 输入通道数
            out_channels=64,    # 输出通道数（卷积核数量）
            kernel_size=3,      # 卷积核大小
            padding=1           # 填充
        )
        self.conv_layer2 = nn.Conv1d(
            in_channels=64,      # 输入通道数
            out_channels=128,    # 输出通道数（卷积核数量）
            kernel_size=3,      # 卷积核大小
            padding=1           # 填充
        )
        self.conv_layer3 = nn.Conv1d(
            in_channels=128,      # 输入通道数
            out_channels=64,    # 输出通道数（卷积核数量）
            kernel_size=3,      # 卷积核大小
            padding=1           # 填充
        )
        self.conv_layer4 = nn.Conv1d(
            in_channels=64,      # 输入通道数
            out_channels=1,    # 输出通道数（卷积核数量）
            kernel_size=3,      # 卷积核大小
            padding=1           # 填充
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc = nn.Linear(4096, 4096)


    def forward(self, x):
        outputs = []
        b, _ = x.shape
        for branch in self.branches:
            x_reshaped = x.view(x.size(0), 1, -1)
            x_conv = branch.conv_net(x_reshaped)
            x_conv = self.leaky_relu(x_conv)
            x_conv = x_conv.view(x.size(0), -1, x_conv.shape[-1])  # 将卷积输出重塑
            output = branch(x_conv)
            output = output[:, 1:, :].transpose(1, 2).reshape(output.shape[0], -1)  # 将输出重塑为4096
            outputs.append(output)

       # 移除每个张量的单一维度
        # outputs = [torch.squeeze(output) for output in outputs]  # 每个张量从 [1, 4096] 变为 [4096]
        # 使用 torch.stack 来沿新的维度堆叠它们
        combined_output = torch.stack(outputs, dim=1)  # 现在形状是 [4, 4096]
        x = self.conv_layer1(combined_output)
        x=self.leaky_relu(x)
        x = self.conv_layer2(x)
        x=self.leaky_relu(x)
        x = self.conv_layer3(x)
        x=self.leaky_relu(x)
        x = self.conv_layer4(x)
        x = torch.squeeze(x, 1)  # 1 表示去掉第二个维度，只在该维度长度为1时压缩
        x = self.fc(x)  # 全连接层，映射到1个输出
        final_outputs=x
        return final_outputs  # 返回四个分支的输出


