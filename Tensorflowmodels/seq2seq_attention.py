import  tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
import random
import copy
def extract_argmax_and_embed(embedding, output_projection=None):
    """
    Get a loop_function that extracts the previous symbol and embeds it. Used by decoder.
    :param embedding: embedding tensor for symbol
    :param output_projection: None or a pair (W, B). If provided, each fed previous output will
    first be multiplied by W and added B.
    :return: A loop function
    """
    def loop_function(prev, _):
        if output_projection is not None:
            prev = tf.matmul(prev, output_projection[0]) + output_projection[1]
        prev_symbol = tf.argmax(prev, 1) #得到对应的INDEX
        emb_prev = tf.gather(embedding, prev_symbol) #得到这个INDEX对应的embedding
        return emb_prev
    return loop_function

class seq2seq_attention_model:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size,hidden_size, is_training,decoder_sent_length=6,
                 initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=5.0,l2_lambda=0.0001):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = initializer
        self.decoder_sent_length=decoder_sent_length
        self.hidden_size = hidden_size
        self.clip_gradients=clip_gradients
        self.l2_lambda=l2_lambda

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")                 #x
        self.decoder_input = tf.placeholder(tf.int32, [None, self.decoder_sent_length],name="decoder_input")  #y, but shift
        self.input_y_label = tf.placeholder(tf.int32, [None, self.decoder_sent_length], name="input_y_label") #y, but shift
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

    def instantiate_weights(self):
        self.Embeding = tf.get_variable("Embedding", [self.vocab_size, self.embed_size],
                                           initializer=self.initializer)
        self.Embedding_label = tf.get_variable("Embedding_label", shape=[self.vocab_size, self.embed_size * 2],
                                               dtype=tf.float32)  # ,initializer=self.initializer
        self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 2, self.vocab_size],
                                            initializer=self.initializer)  # [embed_size,label_size]
        self.b_projection = tf.get_variable("b_projection", shape=[self.vocab_size])

        self.W_z = tf.get_variable([self.embed_size,self.hidden_size],initializer=self.initializer)
        self.U_z = tf.get_variable([self.embed_size,self.hidden_size],initializer=self.initializer)
        self.b_z = tf.get_variable("b_z", shape=[self.hidden_size])
        self.W_r = tf.get_variable([self.embed_size,self.hidden_size],initializer=self.initializer)
        self.U_r = tf.get_variable([self.embed_size,self.hidden_size],initializer=self.initializer)
        self.b_r =  tf.get_variable("b_r", shape=[self.hidden_size])
        self.W_h = tf.get_variable("W_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
        self.U_h = tf.get_variable("U_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
        self.b_h = tf.get_variable("b_h", shape=[self.hidden_size])

        with tf.name_scope("decoder_init_state"):
            self.W_initial_state = tf.get_variable("W_initial_state", shape=[self.hidden_size, self.hidden_size*2], initializer=self.initializer)
            self.b_initial_state = tf.get_variable("b_initial_state", shape=[self.hidden_size*2])

        with tf.name_scope("gru_weights_decoder"):
            self.W_z_decoder = tf.get_variable("W_z_decoder", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.U_z_decoder = tf.get_variable("U_z_decoder", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.C_z_decoder = tf.get_variable("C_z_decoder", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)  # TODO
            self.b_z_decoder = tf.get_variable("b_z_decoder", shape=[self.hidden_size * 2])
            # GRU parameters:reset gate related
            self.W_r_decoder = tf.get_variable("W_r_decoder", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.U_r_decoder = tf.get_variable("U_r_decoder", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.C_r_decoder = tf.get_variable("C_r_decoder", shape=self.hidden_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)  # TODO
            self.b_r_decoder = tf.get_variable("b_r_decoder", shape=[self.hidden_size * 2])

            self.W_h_decoder = tf.get_variable("W_h_decoder", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.U_h_decoder = tf.get_variable("U_h_decoder", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)  # TODO
            self.C_h_decoder = tf.get_variable("C_h_decoder", shape=[self.hidden_size * 2, self.hidden_size * 2],
                                               initializer=self.initializer)
            self.b_h_decoder = tf.get_variable("b_h_decoder", shape=[self.hidden_size * 2])
    def gru_forward(self,embed_words,gru_cell,reverse = False):
        splited_words = tf.split(embed_words,self.sequence_length,axis=1)
        splited_words = [tf.squeeze(words,axis=1) for words in splited_words]
        # 初始化的隐藏状态
        h_t = tf.ones(shape=(self.batch_size,self.embed_size))
        h_t_list = []
        if reverse:
            splited_words = splited_words.reverse()
        for time_step,X_t in enumerate(splited_words):
            h_t = gru_cell(X_t,h_t)
            h_t_list.append(h_t)
        if reverse:
            h_t_list.reverse()
        return h_t_list  # a list,length is sentence_length, each element is [batch_size,hidden_size]





    def gru_cell(self,Xt,ht_minus1):
        """

        :param Xt: shape ->[batch,embed_size]
        :param ht_minus1: ->[batch,embed_size]
        :return:
        """
        # 1.update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(tf.matmul(Xt,self.W_z)+tf.matmul(ht_minus1,self.U_z)+ self.b_z)
        #2.reset gate  controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(tf.matmul(Xt,self.W_r)+ tf.matmul(ht_minus1,self.U_r)+ self.b_r)
        # candidate state
        h_t_candidate= tf.nn.tanh(tf.matmul(Xt,self.W_h))+r_t*(tf.matmul(Xt,self.U_h)+self.b_h)
        return z_t* h_t_candidate + (1-z_t)*ht_minus1
    def gru_cell_decoder(self, Xt, h_t_minus_1,context_vector):
        """
        single step of gru for word level
        :param Xt: Xt:[batch_size,embed_size]
        :param h_t_minus_1:[batch_size,embed_size]
        :param context_vector. [batch_size,embed_size].this represent the result from attention( weighted sum of input during current decoding step)
        :return:

        """
        # 1.update gate: decides how much past information is kept and how much new information is added.
        z_t = tf.nn.sigmoid(
            tf.matmul(Xt, self.W_z_decoder) + tf.matmul(h_t_minus_1, self.U_z_decoder) + tf.matmul(context_vector,
                                                                                                   self.C_z_decoder) + self.b_z_decoder)  # z_t:[batch_size,self.hidden_size]
        # 2.reset gate: controls how much the past state contributes to the candidate state.
        r_t = tf.nn.sigmoid(
            tf.matmul(Xt, self.W_r_decoder) + tf.matmul(h_t_minus_1, self.U_r_decoder) + tf.matmul(context_vector,
                                                                                                   self.C_r_decoder) + self.b_r_decoder)  # r_t:[batch_size,self.hidden_size]
        # candiate state h_t~
        h_t_candiate = tf.nn.tanh(
            tf.matmul(Xt, self.W_h_decoder) + r_t * (tf.matmul(h_t_minus_1, self.U_h_decoder)) + tf.matmul(
                context_vector, self.C_h_decoder) + self.b_h_decoder)  # h_t_candiate:[batch_size,self.hidden_size]
        # new state: a linear combine of pervious hidden state and the current new state h_t~
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate  # h_t:[batch_size*num_sentences,hidden_size]
        return h_t, h_t



    def inference(self):
        self.embed_words = tf.nn.embedding_lookup(self.Embeding,self.input_x)
        #encoder with gru
        hidden_state_forward_list = self.gru_forward(self.embed_words,self.gru_cell)
        hidden_state_backard_list = self.gru_forward(self.embed_words,self.gru_cell,reverse=True)
        hidden_state_final_list = [tf.concat([forward_hidden,backward_hidden],axis=1) for forward_hidden, backward_hidden in zip(hidden_state_forward_list,hidden_state_backard_list)]

        # 3.Decoder using GRU with attention

        thought_vector = tf.stack(hidden_state_final_list,axis=1)# 现在shape是 [batch,seq_len,hidden_size*2]

        initial_state = tf.nn.tanh(
            tf.matmul(hidden_state_backard_list[0], self.W_initial_state) + self.b_initial_state)
        cell = self.gru_cell_decoder
        output_projection=(self.W_projection,self.b_projection) #W_projection:[self.hidden_size * 2, self.num_classes]; b_projection:[self.num_classes]
        loop_function = extract_argmax_and_embed(self.Embedding_label,output_projection) if not self.is_training else None #loop function will be used only at testing, not training.

        attention_state = thought_vector







