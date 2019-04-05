import  torch
import torch.nn as nn

class TextRNN(nn.Module):
    def __init__(self,config):
        super(TextRNN,self).__init__()
        self.vocab_size = config.vocab_size
        self.embed_size = config.embed_size
        self.hidden_size = config.hidden_size

        self.embedding = nn.Embedding(self.vocab_size,self.embed_size)
        self.db = nn.BatchNorm2d(1)
        self.rnn = nn.LSTM(input_size=self.embed_size,
                          hidden_size=config.hidden_size,
                          num_layers=config.num_layers,
                          bias=True,
                          batch_first=True,
                          dropout=config.dropout_rate,
                          bidirectional=True)
        self.linear1 = nn.Linear(2*config.num_layers*config.hidden_size,512)
        self.db2 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512,config.num_classes)

    def forward(self, input):
        self.embed_words = self.embedding(input)
        self.embed_words= self.db(self.embed_words.unsqueeze(1)).squeeze(1)
        outputs,(h_out,c_out) = self.rnn(self.embed_words)
        #hout shape is [2*num_layers,batch_size,hidden_size]
        #outputts shape is [seq_len,batch_layes,hidden_size*2]
        h_out = h_out.transpose(0,1).view(config.batch_size,-1)
        x = self.db2(self.linear1(h_out))
        logits = self.linear2(x)
        return logits







