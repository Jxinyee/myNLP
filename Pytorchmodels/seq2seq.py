import math
import torch
import random
from tqdm import tqdm
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,n_layers=2,drop_out = 0.5):
        super(Encoder,self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hiden_size = hidden_size
        self.n_layers = n_layers
        self.drop_out = drop_out
        self.embeddings = nn.Embedding(vocab_size,embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=drop_out, bidirectional=True)
    def forward(self, input,hidden):
        embed_words = self.embeddings(input)
        if hidden =None:
            hidden = torch.randn(self.n_layers*2,embed_words.size(1),self.hiden_size)
        outputs,hidden = self.gru(embed_words,hidden)
        return outputs,hidden

class Attention(nn.Module):
    def __init__(self,hidden_size):
        super(Attention,self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size*2,hidden_size)
        self.v = nn.Parameter(torch.randn(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
    def forward(self, encoder_outputs,hidden):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1) #shape is {batch,seq_len,H}?
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*seq_len*H]
        attn_logits =self.score(h,encoder_outputs)
        return F.relu(attn_logits,dim=1).unsqueeze(1)#[batch,1,seq_len]


    def score(self,h,encoder_outputs):
        energy =F.softmax(self.attn(torch.cat([h,encoder_outputs],dim =2)))# B ,seq_len, hidden_size
        energy = energy.transpose(1,2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*seq_len]
        return energy.squeeze(1)  # [B*seq_len]

class Decoder(nn.Module):
    def __init__(self,embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder,self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

class Seq2Seq(nn.Module):





















