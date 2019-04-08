#this is a  textrnn model however we use tf to build rnn not use api
import torch
import torch.nn  as nn
import copy
from torch.autograd import Variable
class TextRNN(nn.Module):
    def __init__(self,config):
        super(TextRNN, self).__init__()
        self.vocab_size = config.vocab_size
        self.embed_size = config.embed_size
        self.hidden_size = config.hidden_size
        self.text_len = config.text_len
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.batch_size = config.batch_size
        self.left_first_state = nn.init.uniform(torch.randn(self.batch_size, self.hidden_size), a=0, b=0.1)
        self.right_fist_state = nn.init.uniform(torch.randn(self.batch_size, self.hidden_size), a=0, b=0.1)
        self.W_l1 = nn.Parameter(nn.init.uniform(torch.randn(self.embed_size, self.hidden_size), a=0, b=0.1))
        self.bias_l1 = nn.Parameter(torch.randn(self.hidden_size))
        self.bias_l2 = nn.Parameter(torch.randn(self.hidden_size))
        self.W_l2 = nn.Parameter(nn.init.uniform(torch.randn(self.embed_size, self.hidden_size), a=0, b=0.1))
        self.W_r1 = nn.Parameter(nn.init.uniform(torch.randn(self.embed_size, self.hidden_size), a=0, b=0.1))
        self.W_r2 = nn.Parameter(nn.init.uniform(torch.randn(self.embed_size, self.hidden_size), a=0, b=0.1))
        # for example self.activation = torch.tanh
        self.activation = config.activation


    def forward(self, input):
        self.embed_words = self.embedding(input)
        output = self.birnn(self.embed_words)
    def context_left(self,embedding_previous,context_left_words):
        # in this fuction we can define lstm or gru or normal rnn
        """


        single step of gru for word level copy from webbolg
        :param Xt: embedding_previousXt:[batch_size,embed_size]
        :param h_t_minus_1:context_left_words :[batch_size,embed_size]
        :return:

        # 1.update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1,self.U_z) + self.b_z)  # z_t:[batch_size,self.hidden_size]
        # 2.reset gate: controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1,self.U_r) + self.b_r)  # r_t:[batch_size,self.hidden_size]
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) +r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)  # h_t_candiate:[batch_size,self.hidden_size]
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate  # h_t:[batch_size*num_sentences,hidden_size]
        return h_t
        """

         #normal rnn and we can define different rnn
        left_e =embedding_previous.mm(self.W_l1)
        left_c = context_left_words.mm(self.W_l2)
        left_h= left_c+left_e
        context_left = self.activation(left_h)
        return context_left


    def birnn(self,input):
        embeded_words_split = torch.split(self.embed_words,split_size_or_sections=1,dim=1)#shape[batch,text_len,embed_size]->[seq_len,batch,1,embed_size]
        embeded_words_splited = [torch.squeeze(word,dim=1) for word in embeded_words_split]#shape->[seq_len,batch,embe_words]
        embeding_previous = self.left_first_state
        context_left_previous = torch.zeros(self.batch_size,self.embed_size)
        context_left_list=[]
        for i ,current_words in enumerate(embeded_words_splited):
            context_left = self.context_left(embeding_previous,context_left_previous)
            context_left_list.append(context_left)
            embeding_previous = current_words
            context_left_previous = context_left

        embeding_right_previous= self.right_fist_state
        context_right_previous = torch.zeros(self.batch_size,self.embed_size)
        context_right_list=[]

        embeded_words_splited_right = copy.copy(embeded_words_splited)
        embeded_words_splited_right.reverse()
        for i ,current_words in enumerate(embeded_words_splited_right):
            context_left = self.context_right(embeding_right_previous,context_right_previous)
            context_right_list.append(context_left)
            embeding_right_previous = current_words
            context_right_previous = context_left

        output_list = []
        for index, current_embedding_word in enumerate(embeded_words_splited):
            representation = torch.cat([context_left_list[index], current_embedding_word, context_right_list[index]],
                                       dim=1)
            # print(i,"representation:",representation)
            output_list.append(representation)  # shape:sentence_length个[None,embed_size*3]
        # 5. stack list to a tensor
        # print("output_list:",output_list) #(3, 5, 8, 100)
        output = torch.stack(output_list, dim=1)  # shape:[None,sentence_length,embed_size*3]
        # print("output:",output)
        return output












