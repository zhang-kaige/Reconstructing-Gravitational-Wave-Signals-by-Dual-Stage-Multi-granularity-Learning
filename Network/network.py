import torch
import torch.nn as nn
from Transformer import Transformer
import time

class N(nn.Module):
    def __init__(self, dim1, dim2, dim3, depth, heads, mlp_dim):
        super(N, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3

        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=dim1, kernel_size=64, stride=64, padding=0)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=dim2, kernel_size=128, stride=128, padding=0)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=dim3, kernel_size=256, stride=256, padding=0)

        self.trans1 = Transformer(dim1, depth, heads, dim1 // heads, mlp_dim, dropout=0.)
        self.trans2 = Transformer(dim2, depth, heads, dim2 // heads, mlp_dim, dropout=0.)
        self.trans3 = Transformer(dim3, depth, heads, dim3 // heads, mlp_dim, dropout=0.)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, strain: torch.Tensor):
        strain = strain.view(strain.size(0), 1, -1)

        token_8 = self.conv1(strain)
        token_8 = token_8.permute(0, 2, 1)
        token_8= self.leaky_relu(token_8)

        token_16 = self.conv2(strain)
        token_16 = token_16.permute(0, 2, 1)
        token_16= self.leaky_relu(token_16)

        token_32 = self.conv3(strain)
        token_32 = token_32.permute(0, 2, 1)
        token_32= self.leaky_relu(token_32)

        output_8, output_16, output_32 = self.trans1(token_8), self.trans2(token_16), self.trans3(token_32)
        return output_8, output_16, output_32

class N1(nn.Module):
    def __init__(self, dim1, dim2, dim3, depth, heads, mlp_dim):
        super(N1, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3

        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=dim1, kernel_size=64, stride=64, padding=0)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=dim2, kernel_size=128, stride=128, padding=0)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=dim3, kernel_size=256, stride=256, padding=0)
        self.trans1 = Transformer(dim1, depth, heads, dim1 // heads, mlp_dim, dropout=0.)
        self.trans2 = Transformer(dim2, depth, heads, dim2 // heads, mlp_dim, dropout=0.)
        self.trans3 = Transformer(dim3, depth, heads, dim3 // heads, mlp_dim, dropout=0.)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, signal: torch.Tensor, strain: torch.Tensor):

        signal = signal.view(signal.size(0), 1, -1)
        strain = strain.view(strain.size(0), 1, -1)
        
        signal_8 = self.conv1(signal)
        signal_8 = signal_8.permute(0, 2, 1) 
        signal_8= self.leaky_relu(signal_8)

        signal_16 = self.conv2(signal)
        signal_16 = signal_16.permute(0, 2, 1) 
        signal_16= self.leaky_relu(signal_16)

        signal_32 = self.conv3(signal)
        signal_32 = signal_32.permute(0, 2, 1) 
        signal_32= self.leaky_relu(signal_32)

        strain_8 = self.conv1(strain)
        strain_8 = strain_8.permute(0, 2, 1)
        strain_8= self.leaky_relu(strain_8)

        strain_16 = self.conv2(strain)
        strain_16 = strain_16.permute(0, 2, 1) 
        strain_16= self.leaky_relu(strain_16)

        strain_32 = self.conv3(strain)
        strain_32 = strain_32.permute(0, 2, 1) 
        strain_32= self.leaky_relu(strain_32)

        token_8, token_16, token_32  = torch.cat((signal_8, strain_8), dim=1), torch.cat((signal_16, strain_16), dim=1), torch.cat((signal_32, strain_32), dim=1)
        output_8, output_16, output_32 = self.trans1(token_8), self.trans2(token_16), self.trans3(token_32)
        return output_8[:,:64,:], output_16[:,:32,:], output_32[:,:16,:]


class N2(nn.Module):
    def __init__(self, dim1, dim2, dim3, depth, heads, mlp_dim, use_strain=True):
        super(N2, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=dim1, kernel_size=64, stride=64, padding=0)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=dim2, kernel_size=128, stride=128, padding=0)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=dim3, kernel_size=256, stride=256, padding=0)

        self.conv4 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.trans1 = Transformer(dim1, depth, heads, dim1 // heads, mlp_dim, dropout=0.)
        self.trans2 = Transformer(dim2, depth, heads, dim2 // heads, mlp_dim, dropout=0.)
        self.trans3 = Transformer(dim3, depth, heads, dim3 // heads, mlp_dim, dropout=0.)

        self.use_strain = use_strain

        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.bn4 = nn.BatchNorm1d(num_features=64)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, feature, strain):
        strain = strain.view(strain.size(0), 1, -1)
        feature_8, fearture_16, feature_32 = feature[0], feature[1], feature[2]

        strain_8 = self.conv1(strain)
        strain_8 = strain_8.permute(0, 2, 1)  
        strain_8= self.leaky_relu(strain_8)

        strain_16 = self.conv2(strain)
        strain_16 = strain_16.permute(0, 2, 1)  
        strain_16= self.leaky_relu(strain_16)

        strain_32 = self.conv3(strain)
        strain_32 = strain_32.permute(0, 2, 1)  
        strain_32= self.leaky_relu(strain_32)

        token_8, token_16, token_32  = torch.cat((feature_8, strain_8), dim=1), torch.cat((fearture_16, strain_16), dim=1), torch.cat((feature_32, strain_32), dim=1)
        output_8, output_16, output_32 = self.trans1(token_8), self.trans2(token_16), self.trans3(token_32)
        output_8 = output_8.flatten(1).unsqueeze(dim=1)
        output_16 = output_16.flatten(1).unsqueeze(dim=1)
        output_32 = output_32.flatten(1).unsqueeze(dim=1)
        if self.use_strain == True:
            output_8 = output_8[:,:,4096:]
            output_16 = output_16[:,:,4096:]
            output_32 = output_32[:,:,4096:]
        else:
            output_8 = output_8[:,:,4096:]
            output_16 = output_16[:,:,4096:]
            output_32 = output_32[:,:,4096:]
        output = torch.cat((output_8, output_16, output_32), dim=1)

        # with torch.cuda.amp.autocast(enabled=False):
        #3-64
        x = self.conv4(output)
        # ori_x = x
        x = self.bn1(x)
        x=self.leaky_relu(x)
        # x = self.max_pool(x)

        #64-128
        x = self.conv5(x)
        x = self.bn2(x)
        x=self.leaky_relu(x)
        # x = self.max_pool(x)

        #128-256
        # x = self.conv8(x)
        # # x = self.bn3(x)
        # x=self.leaky_relu(x)
        # x = self.max_pool(x)
        
        #256-512
        # x = self.conv10(x)
        # x=self.leaky_relu(x)
        #512-256
        # x = self.conv11(x)
        # x=self.leaky_relu(x)

        #256-128
        # x = self.conv9(x)
        # # x = x + ori_x_1
        # x=self.leaky_relu(x)
        # x = self.max_pool(x)

        #128-64
        x = self.conv6(x)
        # x = x + ori_x
        x = self.bn4(x)
        x=self.leaky_relu(x)

        #64-1
        x = self.conv7(x)

        return x
    
# if __name__ == '__main__':
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     signal = torch.randn(1,1,4096).to(device)
#     strain = torch.randn(1,1,4096).to(device)
#     N1_model = N1(dim=512, depth=4, heads=8, dim_head=64, mlp_dim=2048).to(device)
#     N_model = N(dim=512, depth=4, heads=8, dim_head=64, mlp_dim=2048).to(device)
#     N2_model = N2(dim=512, depth=4, heads=8, dim_head=64, mlp_dim=2048).to(device)
#     output = N1_model(signal, strain)
#     y = N_model(strain)
#     y2 = N2_model(signal, strain) 
#     print("ok")