import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import torch.optim as optim

class CifarModel(nn.Module):
    def __init__(self,hidden_dim,num_classes):
        super().__init__()
        self.l1 = nn.Linear(32 * 32 * 3, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    
class LoRALinear(nn.Module):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rank: int = 0, 
        lora_alpha: int = 1, 
    ):
        super().__init__()
        
        self.lora_alpha = lora_alpha
        self.rank = rank
        # Actual trainable parameters
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.empty(rank, out_features))
        # according to original paper
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        self.scaling = self.lora_alpha / self.rank

    def forward(self,x):
        out = x @ (self.A @ self.B)
        return out
    

class CifarLoRAModel(CifarModel):
    def __init__(self,hidden_dim,num_classes,rank, alpha):
        super().__init__(hidden_dim,num_classes)
        self.rank = rank
        self.alpha = alpha
        for name, parameter in self.named_parameters():
            parameter.requires_grad = False
        self.l1_lora = LoRALinear(32 * 32 * 3, hidden_dim, self.rank)
        self.l2_lora = LoRALinear(hidden_dim, hidden_dim, self.rank)
        self.l3_lora = LoRALinear(hidden_dim, num_classes, self.rank)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.l1(x) + self.alpha * self.l1_lora(x))
        x = F.relu(self.l2(x) + self.alpha * self.l2_lora(x))
        x = self.l3(x) + self.alpha * self.l3_lora(x)
        return x
