# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from mymodel.nn_model_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
import osada


# from raytracing import RayTracing


## =====================================================================
## 


class MultiplyLayer(nn.Module):


    def __init__(self):
        super(MultiplyLayer, self).__init__()


    def forward(self, A, B):
        return A * B


class SumLayer(nn.Module):
    def __init__(self):
        super(SumLayer, self).__init__()

    def forward(self, x):
        # xの要素の総和を求めます
        return x.sum(dim=1, keepdim=True)
    

class RefMapIndentityLayer(nn.Module):
    def __init__(self, name='refmap_identity_layer'):
        super(RefMapIndentityLayer, self).__init__()
        self.name = name

    def forward(self, x):
        return x


## =====================================================================
## PointNet++


## https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master

class PointNetPPEncoderDecoder(nn.Module):
    def __init__(self, ):
        super(PointNetPPEncoderDecoder, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.2) # 0.5
        

        ## =====================================================================
        ## modified
        
        # self.conv2 = nn.Conv1d(128, 64, 5, padding=2)
        self.conv2 = nn.Conv1d(128, 1, 3, padding=1)
        
        self.drop2 = nn.Dropout(0.2)
        
        # self.weighted_sum_1 = nn.Conv1d(64, 16, 5, padding=2)
        self.weighted_sum_2 = nn.Conv1d(16, 1, 3, padding=1)
        # self.bn_after_1 = nn.BatchNorm1d(64)
        self.bn_after_2 = nn.BatchNorm1d(16)
        self.bn_after_3 = nn.BatchNorm1d(1)
        
        self.multiply_layer = MultiplyLayer()
        self.refmap_identity_layer = RefMapIndentityLayer(name='refmap_identity_layer')
        self.fc_to_I_C = SumLayer()


    def forward(self, xyz, point_wise_coef):
        
        ## xyz (a, b, c) -> (a, c, b)
        xyz = xyz.permute(0, 2, 1)
        
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        
        ## =====================================================================
        ## modified
        
        # x = F.relu(x)

        # x = self.bn_after_1(x)
        # x = self.weighted_sum_1(x)
        # x = F.relu(x)
        
        # x = self.bn_after_2(x)
        # x = self.drop2(x)
        # x = self.weighted_sum_2(x)
        x = self.bn_after_3(x)
        
        x = F.sigmoid(x)
        
        # x = self.multiply_layer(x, 2.)
        
        ## xyz (a, b, c) -> (a, c, 1)
        # x = x.permute(0, 2, 1).squeeze(2)
        
        refmap = self.refmap_identity_layer(x)
        
        ## xyz (a, b) -> (a, 1, b)
        point_wise_coef = point_wise_coef.unsqueeze(1)
        
        x = self.multiply_layer(refmap, point_wise_coef)
        x = F.relu(x)
        ## xyz (a, 1, b) -> (a, b)
        x = x.squeeze(1)
        
        x = self.fc_to_I_C(x)
        
        return x
        # return x, refmap
        # return x, l4_points
    
    
class PointNetPPClassifier(nn.Module):
    def __init__(self, normal_channel=True):
        super(PointNetPPClassifier, self).__init__()
        """
        [!] deprecated
        """
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        ## modified
        self.fc3 = nn.Linear(256, 512)
        
        ## =====================================================================
        ## modified
        self.multiply_layer = MultiplyLayer()
        self.refmap_identity_layer = RefMapIndentityLayer(name='refmap_identity_layer')
        self.fc_to_I_C = SumLayer()
        
        
    def forward(self, xyz, point_wise_coef):
        
        ## xyz (a, b, c) -> (a, c, b)
        xyz = xyz.permute(0, 2, 1)
        
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        # x = F.log_softmax(x, -1)
        
        ## =====================================================================
        ## modified
        x = F.sigmoid(x)
        # x = F.relu(x)
        
        x = self.refmap_identity_layer(x)
        
        x = self.multiply_layer(x, point_wise_coef)
        x = F.relu(x)
        x = self.fc_to_I_C(x)        


        return x
        # return x,l3_points

        
        
## =====================================================================
## transformer (tmp)


class MyTransformerLayer(nn.Module):

    def __init__(self, input_dim, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(MyTransformerLayer, self).__init__()
        self.input_dim = input_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        ## imegenet
        

    def forward(self, src):
        # Transformerは入力の形状が [seq_len, batch_size, input_dim] であることを期待しているので、調整します
        src = src.permute(1, 0, 2)  # [batch_size, seq_len, input_dim] -> [seq_len, batch_size, input_dim]
        output = self.transformer_encoder(src)
        return output.permute(1, 0, 2)  # 元の形状に戻します


class SimpleTransformerModel(nn.Module):

    def __init__(self, length, device=None):
        super(SimpleTransformerModel, self).__init__()
        
        self.length = length
        if device is None: device = torch.device('cuda')
        
        # Transformer層のパラメータ
        self.transformer = MyTransformerLayer(input_dim=3, nhead=3, num_encoder_layers=6, 
                                            dim_feedforward=2048, dropout=0.1).to(device)
        self.fc_to_I_C = SumLayer()  # Transformerの出力を元のサイズに合わせて調整します
        
        self.multiply_layer = MultiplyLayer()
        self.refmap_identity_layer = RefMapIndentityLayer(name='refmap_identity_layer')


    def forward(self, x, point_wise_coef):
        # 入力xの形状：[batch_size, length, 6]
        
        xyz, rgb = x.split(3, dim=2)  # xを3次元座標とRGBに分割します
        
        x = self.transformer(x)  # Transformer層を適用します
        
        x = torch.sigmoid(x)
        x = torch.mean(x, dim=2)
        
        x = self.refmap_identity_layer(x)
        
        x = self.multiply_layer(x, point_wise_coef)
        x = F.relu(x)
        x = self.fc_to_I_C(x)
        
        return x

    
## =====================================================================
## tmp


class SimpleLinearModel(nn.Module):
    
    
    def __init__(self, length):
        super(SimpleLinearModel, self).__init__()
        
        self.length = length
        
        # self.fc1 = nn.Linear(length * 6, 512)
        self.fc1 = nn.Linear(length * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc_final = nn.Linear(64, length)
        self.fc_to_I_C = SumLayer()
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn_final = nn.BatchNorm1d(length)
        
        self.dropout = nn.Dropout(0.2)
        
        self.multiply_layer = MultiplyLayer()
        self.refmap_identity_layer = RefMapIndentityLayer(name='refmap_identity_layer')
        

    def forward(self, x, point_wise_coef):

        x = x.view(x.size(0), -1)  # バッチサイズ x (N * 6)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        
        x = self.dropout(x)
        x = F.relu(self.fc_final(x))
        x = self.bn_final(x)
        
        x = F.sigmoid(x)
        
        x = self.refmap_identity_layer(x)
        
        x = self.multiply_layer(x, point_wise_coef)
        x = F.relu(x)
        x = self.fc_to_I_C(x)
        
        return x


## =====================================================================
## backbone master class


class Backbone:
    
    
    def __init__(self, input_shape, output_shape=None, model_type='SimpleLinear', lr=1e-3,
                 is_load_weight=False, weight_path=None, device=None):
        
        self.input_shape = input_shape
        # self.output_shape = input_shape if output_shape is None else output_shape
        self.model_type = model_type
        
        self.length = input_shape[0]
        
        if device is None: device = torch.device('cuda')
        self.device = device
        
        self.build_model(lr=lr)
        self.model.to(self.device)
        
        if is_load_weight:
            self.model.load_state_dict(torch.load(weight_path))
            osada.cprint(f'@ loaded weight from {weight_path}', 'cyan')
    
    
    def build_model(self, lr):
        
        if self.model_type == 'SimpleLinear':
            self.model = SimpleLinearModel(self.length)
        
        elif self.model_type == 'SimpleTransformer':
            self.model = SimpleTransformerModel(self.length, device=self.device)
        
        elif self.model_type == 'PointNet++':
            self.model = PointNetPPEncoderDecoder()
        
        else:
            raise ValueError('\
Hey, you have to choose a model type from "SimpleLinear" or "SimpleTransformer" or "PointNet"!')
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    
    def __str__(self) -> str:
        
        return f'Backbone: {self.model_type}, {self.input_shape} -> {self.output_shape}'


## =====================================================================
## hidden layer extracted model


class HiddenLayerExtractedModel(nn.Module):
    def __init__(self, original_model, layer_name='refmap_identity_layer'):
        super(HiddenLayerExtractedModel, self).__init__()
        self.original_model = original_model
        self.layer_name = layer_name
        self.output = None

    def forward(self, x, point_wise_coef):
        # output = None
        def get_layer_output(module, input, output):
            self.output = output
            # return output
        
        hook = getattr(self.original_model, self.layer_name).register_forward_hook(get_layer_output)
        
        _ = self.original_model(x, point_wise_coef)
        
        hook.remove()
        
        return self.output
    
