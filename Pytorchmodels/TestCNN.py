import  torch
import  torch.nn.functional as F
import  torch.nn as nn
class TextCNN(nn.Module):
    def __init__(self,config):
        super(self,TextCNN).__init__()
        self.is_Training = config.is_traning
        self.vocab_size = config.vocab_size
        self.embed_size = config.embed_size
        self.num_filters = config.num_filters
        self.num_classes = config.num_classes
        self.is_dropout = config.is_dropout
        self.embedings = nn.Embedding(self.vocab_size,self.embed_size)

        #singer cnn layer
        self.singerconvs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.embed_size,kernel_size=h,out_channels=self.num_filters),
                          nn.BatchNorm1d(num_features=self.num_filters),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_text_len-h+1))#shape [batch,embed_size,text_len]->[batch,num_filters,text_len-h+1]->[batch,num_filters,1]
            for h in config.filter_sizes#filter_size can be[3,4,5] 代表着 上下文的窗口
        ])
        # multicnnlayer
        '''
        this may need more memory
        self.multicnns = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.embed_size,kernel_size=h,out_channels=self.num_filters),
                          nn.BatchNorm1d(num_features=self.num_filters),
                          nn.ReLU(),
                          nn.Conv1d(in_channels=self.num_filters,out_channels=self.num_filters,kernel_size=h),
                          nn.BatchNorm1d(num_features=self.num_filters),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_text_len-2*h+2))
            for h in config.filter_sizes
        ])
        
        '''

        self.fc =nn.Linear(self.num_filters*config.filter_sizes,self.num_classes)
        if os.path.exists(config.embedding_path) and config.is_training and config.is_pretrain:
            print("Loading pretrain embedding...")
            self.embedding.weight.data.copy_(torch.from_numpy(np.load(config.embedding_path)))

    def forward(self, input):
        embed_words = self.embedings(input)
        embed_words = embed_words.permute(0,2,1)#[batch,text_len,embed_size]->[batch,embed_size,text_len]
        out = [conc(embed_words) for conv in self.singerconvs]

        out = torch.cat(out,dim = 1)
        out = out.view(-1,out.size(1))
        if not self.is_dropout:
            out = F.dropout(input=out, p=self.dropout_rate)
            out = self.fc(out)
        return out

    def get_optimizer(self,lr1,lr2,weight_decay):
        return torch.optim.Adam([{'params':self.singerconvs.parameters()},
                                 {'params':self.fc.parameters()},
                                 {'params':self.embedings.parameters(),'lr':lr1}
        ],lr=lr2,weight_decay= weight_decay)




