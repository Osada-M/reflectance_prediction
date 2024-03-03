# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from mymodel.nn_model_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
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


class PointNet2(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(PointNet2, self).__init__()
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
        self.fc3 = nn.Linear(256, num_class)
        
        self.refmap_identity_layer = RefMapIndentityLayer(name='refmap_identity_layer')

    def forward(self, xyz):
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
        x = F.log_softmax(x, -1)


        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
        
        
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

    def __init__(self, length):
        super(SimpleTransformerModel, self).__init__()
        
        self.length = length
        
        # Transformer層のパラメータ
        self.transformer = MyTransformerLayer(input_dim=6, nhead=6, num_encoder_layers=6, 
                                            dim_feedforward=2048, dropout=0.1)
        self.fc_to_I_C = SumLayer()  # Transformerの出力を元のサイズに合わせて調整します
        
        self.multiply_layer = MultiplyLayer()
        self.refmap_identity_layer = RefMapIndentityLayer(name='refmap_identity_layer')


    def forward(self, x, point_wise_coef):
        # 入力xの形状：[batch_size, length, 6]
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
        
        self.fc1 = nn.Linear(length * 6, 512)
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
                 is_load_weight=False, weight_path=None):
        
        self.input_shape = input_shape
        # self.output_shape = input_shape if output_shape is None else output_shape
        self.model_type = model_type
        
        self.length = input_shape[0]
        
        self.build_model(lr=lr)
        
        if is_load_weight:
            self.model.load_state_dict(torch.load(weight_path))
            osada.cprint(f'@ loaded weight from {weight_path}', 'cyan')
    
    
    def build_model(self, lr):
        
        if self.model_type == 'SimpleLinear':
            self.model = SimpleLinearModel(self.length)
        
        elif self.model_type == 'SimpleTransformer':
            self.model = SimpleTransformerModel(self.length)
        
        elif self.model_type == 'PointNet':
            self.model = PointNet2(num_class=1, normal_channel=True)
        
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
    
