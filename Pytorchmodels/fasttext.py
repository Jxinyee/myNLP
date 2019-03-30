import  numpy as np
from torch.autograd import Variable
import torch
from torch import nn
import  tensorflow as tf
#用tensorflow和pytorch俩种方法试着实现
class Fasttextpy(nn.Module):
    #这里的config我们可以定一个config结构 去传入
    def __init__(self,config):
        self.embedding =nn.Embedding(config.vocab_size,config.embedding_dim)
        self.linear = nn.Linear(config.embedding_dim,config.num_class)
    def forward(self, input):
        self.embeded = self.embedding(input)
        text_embeded = torch.mean(self.embeded,dim=1)
        logits = self.linear(text_embeded)
        return logits


