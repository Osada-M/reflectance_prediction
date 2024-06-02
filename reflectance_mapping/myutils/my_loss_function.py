import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class ErrorOfCurrent:
    
    def __init__(self) -> None:
        pass        
    
    
    @staticmethod
    def __call__(outputs, labels):
        
        return torch.mean((outputs - labels)**2)


## =====================================================================
## 


class LossForBackbone(nn.Module):
    

    def __init__(self):
        
        super(LossForBackbone, self).__init__()


    def forward(self, input, target):
        
        # input = input.unsqueeze(1).expand_as(self.user_defined_value)
        # target = target.unsqueeze(1).expand_as(self.user_defined_value)
        
        # return (target - input) * self.user_defined_value
        
        losses = torch.abs(target - input) * self.user_defined_value.mean()
        loss = losses.mean()
        
        return loss

    
    def set_user_defined_value(self, user_defined_value):
        
        self.user_defined_value = user_defined_value


## =====================================================================
## 


class MSE(nn.Module):
    
    def __init__(self):
        
        super(MSE, self).__init__()
        
        self.loss = nn.MSELoss()
        
    def forward(self, input, target):
        
        return self.loss(input, target)


class NLL(nn.Module):
    def __init__(self):
        super(NLL, self).__init__()

    def forward(self, pred, target, trans_feat=None):
        total_loss = F.nll_loss(pred, target)

        return total_loss