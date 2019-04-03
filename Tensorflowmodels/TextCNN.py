import numpy as np
import tensorflow as tf
# learning from CS 224N
#TextCNN 的步骤: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.softmax layer.
class TextCNN:
    def __init__(self,config):
        #set config
        config.initializer =tf.random_normal_initializer(stddev=0.1)
        config.decay_rate = 0.9
        config.clip_gradients = 5
        config.decay_steps = 400
        config.l2_lambda = 0.05
        config.use_multicnnlayers = False
        self.num_classes = config.num_classes
        self.batch_size = config.batch_size
        self.sequence_len = config.sequnce_len
        self.vocab_size = config.vocab_size
        self.embed_size = config.embed_size
        self.use_multicnn = config.use_multicnnlayers
        self.learning_rate = tf.Variable(config.learning_rate,trainable=False,name="learning_rate")
        self.learning_rate_decay_half_op =tf.assign(self.learning_rate,self.learning_rate*config.decay_rate)
        self.filter_sizes =config.filter_sizes
        self.num_filters = config.num_filters
        self.initializer = config.initializer
        self.l2_lambda = config.l2_lambda
        self.num_filters_total = self.num_filters * len(config.filter_sizes)  # how many filters totally.

        self.clip_gradients = config.clip_gradients
        # 之所以要设置is_training_flag  是因为 我们在写代码的时候使用了dorpout 方法

        self.is_training_flag = tf.placeholder(tf.bool, name="is_training_flag")

        # add placeholder
        self.input_x =tf.placeholder(tf.int32,[None,self.sequence_len],name="input_x")
        self.y_label =tf.placeholder(tf.int32,[None,self.num_classes],name="y_label")
        self.drop_keep_prob = tf.placeholder(tf.float32,name="drop_prob")

        #初始化的时候应该尽量让参数小一些，这一点很重要
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.b1 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.b2 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.decay_steps, self.decay_rate = config.decay_steps, config.decay_rate
        self.init_weights()
        self.logits = self.inference()
        self.loss = self.loss_multilabel()
        self.train_op = self.train()



    def init_weights(self):
        self.embeddings  = tf.get_variable([self.vocab_size,self.embed_size],name="embedings")
        self.W_projection = tf.get_variable([self.num_filters_total,self.num_classes],name="softmax_weight")
        self.bias = tf.get_variable([self.num_classes],name="softmax_bias")
    def inference(self):
        self.embed_words = tf.nn.embedding_lookup(self.embeddings,self.input_x)# 注意此时的shape是[None,self.sequence_len,self.embed_size]

        self.embed_words_expanded = tf.expand_dims(self.embed_words,-1) # 这时候shape是[None,self.sequence_len,self.embed_size,1]
        #注意我们之所以这么做是因为在tensorflow中当我们调用nn.conv2D 时候 self.embed_words_expanded 可以看成一组图像数据[batch,height,width,channel]
        if self.use_multicnn:
            h = self.multi_cnn_layer()
        else:
            h =self.singer_cnn_layer()
        with tf.name_scope("output"):
            logits = tf.matmul(h,self.W_projection)+self.bias

        return  logits

    def singer_cnn_layer(self):
        out_pooled = []
        for i ,filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("cnn filter_size %d "%filter_size):
                # 事实上我们随机生成的filter 也是可以作为参数进行梯度下降，调整的，但是CS224N上面是自己设置的 比如在这里 我们可以设置
                #filter1 是tf.ones(filter_size,self.embed_size,1,1) filter2 是tf.randn(....) filter3 是.... 最后concat 就可以了,
                filter = tf.get_variable("filter_size is  %d"%filter_size,[filter_size,self.embed_size,1,self.num_filters],initializer=self.initializer)
                conv = tf.nn.conv2d(self.embed_words_expanded,filter,strides=[1,1,1,1],padding="Valid",name="conv")#shape =# shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv = tf.layers.batch_normalization(conv,training=self.is_training_flag,name="conv_bn")
                #add bias
                b = tf.get_variable(self.num_filters,initializer=self.initializer)
                # 用非线性层 可以使relu 也可以是tanh
                #h = tf.nn.tanh(tf.nn.bias_add(conv,b),name="tanh")
                h = tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")#shape =# shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                pooled = tf.nn.max_pool(h,ksize=[1,self.sequence_len-filter_size+1,1,1],padding="VALID,strides=[1,1,1,1],name="pool")#shape =[batch,1,1,num_filters]
                out_pooled.append(pooled)
        self.h_out = tf.concat(out_pooled,axis=3)

        self.h_out_flat = tf.reshape(self.h_out,[-1,self.num_filters_total])
        # 使用dropout
        with tf.name_scope("dropout"):
            h_drop =tf.nn.dropout(self.h_out_flat,keep_prob=self.drop_keep_prob)
        h = tf.layers.dense(h_drop,self.num_filters_total,activation=tf.nn.tanh,use_bias=True)
        return h
    #事实上 我们在做卷积的时候可能卷积层 不止一层,下面是俩层CNN的设计思路
    def multi_cnn_layer(self):
        out_pooled = []
        for i ,filter_size in enumerate(self.filter_sizes):

            with tf.variable_scope("cnn filter_size %d "%filter_size):
                # 事实上我们随机生成的filter 也是可以作为参数进行梯度下降，调整的，但是CS224N上面是自己设置的 比如在这里 我们可以设置
                #filter1 是tf.ones(filter_size,self.embed_size,1,1) filter2 是tf.randn(....) filter3 是.... 最后concat 就可以了,
                filter = tf.get_variable("filter1_size is  %d"%filter_size,[filter_size,self.embed_size,1,self.num_filters],initializer=self.initializer)
                conv1 = tf.nn.conv2d(self.embed_words_expanded,filter,strides=[1,1,1,1],padding="VALID",name="conv")#shape =# shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv1= tf.layers.batch_normalization(conv1,training=self.is_training_flag,name="conv_bn")
                #add bias
                b = tf.get_variable(self.num_filters,initializer=self.initializer)
                # 用非线性层 可以使relu 也可以是tanh
                #h = tf.nn.tanh(tf.nn.bias_add(conv,b),name="tanh")
                h = tf.nn.relu(tf.nn.bias_add(conv1,b),name="relu")#shape =# shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                h = tf.reshape(h,[-1,self.sequence_len-filter_size+1,self.num_filters,1])# shape:[batch_size,sequence_length - filter_size + 1,num_filters,1]
                filter2 = tf.get_variable("filter2_size is  %d"%filter_size,[filter_size,self.num_filters,1,self.num_filters])#shape :[batch,sequence_len-2*filter_size+2,1,num_filters]
                conv2 = tf.nn.conv2d(h,filter2,strides=[1,1,1,1],padding="Valid",name="conv2")
                b2 = tf.get_variable(self.num_filters,initializer=self.initializer)
                h = tf.nn.relu(tf.nn.bias_add(conv2,b2),name="relu2")
                pooled = tf.nn.max_pool(h,ksize=[1,self.sequence_len-2*filter_size+2,1,1],padding="VALID",strides=[1,1,1,1],name="pool")
                out_pooled.append(pooled)
        h_out = tf.concat(out_pooled,axis=3)
        h_out_flat = tf.reshape(h_out,[-1,self.num_filters_total])
        with tf.name_scope("dropout"):
            h_drop =tf.nn.dropout(h_out_flat,keep_prob=self.drop_keep_prob)
        h = tf.layers.dense(h_drop,self.num_filters_total,activation=tf.nn.tanh,use_bias=True)
        return h

    def loss_multilabel(self, l2_lambda=0.0001):  # 0.0001#this loss function is for multi-label classification
        with tf.name_scope("loss"):

            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_label,
                                                             logits=self.logits)  # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:", losses)  # shape=(?, 1999).
            losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
            loss = tf.reduce_mean(losses)  # shape=().   average loss in the batch
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss
    def train(self):
        learning_rate =tf.train.exponential_decay(self.learning_rate,self.global_step,self.decay_steps,self.decay_rate,staircase=True)
        self.learning_rate = learning_rate
        optim = tf.train.AdamOptimizer(learning_rate)
        gradients,variables = zip(*optim.compute_gradients(self.loss))
        gradients ,_ = tf.clip_by_global_norm(gradients)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  #ADD 2018.06.01
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op












































