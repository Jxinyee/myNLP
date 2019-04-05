import tensorflow as tf
import tensorflow.contrib.rnn as rnn
class TextRNN:
    def __init__(self,config):
        config.initializer = tf.random_normal_initializer(stddev=0.1)
        config.decay_rate = 0.9
        config.clip_gradients = 5
        config.decay_steps = 400
        config.l2_lambda = 0.05
        config.use_multirnnlayers = False
        self.num_classes = config.num_classes
        self.batch_size = config.batch_size
        self.sequence_len = config.sequnce_len

        self.vocab_size = config.vocab_size
        self.embed_size = config.embed_size
        self.use_multicnn = config.use_multicnnlayers
        self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * config.decay_rate)
        self.filter_sizes = config.filter_sizes
        self.num_filters = config.num_filters
        self.initializer = config.initializer
        self.hidden_size = self.embed_size
        self.l2_lambda = config.l2_lambda


        #add placeholder
        self.is_training_flag = tf.placeholder(tf.bool, name="is_training_flag")

        # add placeholder
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_len], name="input_x")
        self.y_label = tf.placeholder(tf.int32, [None, self.num_classes], name="y_label")
        self.drop_keep_prob = tf.placeholder(tf.float32, name="drop_prob")

        # 初始化的时候应该尽量让参数小一些，这一点很重要
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.init_weights()
        self.logits = self.inference()
        self.loss = self.loss_multilabel()
        self.train_op = self.train()

    def init_weights(self):
        with tf.name_scope("embeddings"):
        self.embeddings =tf.get_variable("embedding",shape=[self.vocab_size,self.embed_size],initializer=self.initializer)
        self.W = tf.get_variable("w_projection",shape=[self.hidden_size*2,self.num_classes],initializer= self.initializer)
        self.b = tf.get_variable("bias",shape=[self.num_classes],initializer=self.initializer)
    def inference(self):
        self.embed_words = tf.nn.embedding_lookup(self.embeddings,self.input_x)
        if self.use_multicnn:
            h =self.multilstm()
        else:
            h =self.singerlstm()
        with tf.name_scope("output"):
            logits = tf.matmul(h,self.W)+self.b

        return  logits

    def singerlstm(self):
        # we just uses bi_lstm ->fc layers -> softmax
        lstm_forward = rnn.BasicLSTMCell(num_units=self.embed_size)
        lstm_backward =  rnn.BasicLSTMCell(num_units=self.embed_size)
        if self.drop_keep_prob is not None:
            lstm_backward =rnn.DropoutWrapper(lstm_backward,output_keep_prob=self.dropout_keep_prob)
            lstm_forward = rnn.DropoutWrapper(lstm_forward,output_keep_prob=self.drop_keep_prob)
        outputs,state = tf.nn.bidirectional_dynamic_rnn(lstm_forward,lstm_backward,self.embed_words)
        #fw_out,bk_out = outputs
        output_rnn = tf.concat(outputs,axis=2)#shape = [batch,seq_len,self.embed_size*2]
        # the structure is in img we just get the last seq
        output_rnn= output_rnn[:,-1,:]#[batch,embed_size*2]
        #fc layer
        return output_rnn

    def multilstm(self):
        lstm_forward = rnn.BasicLSTMCell(num_units=self.embed_size)
        lstm_backward = rnn.BasicLSTMCell(num_units=self.embed_size)
        if self.drop_keep_prob is not None:
            lstm_backward =rnn.DropoutWrapper(lstm_backward,output_keep_prob=self.dropout_keep_prob)
            lstm_forward = rnn.DropoutWrapper(lstm_forward,output_keep_prob=self.drop_keep_prob)
        outputs,state = tf.nn.bidirectional_dynamic_rnn(lstm_forward,lstm_backward,self.embed_words)#out_put_shape = [2,batch,seq_len,seq_len,embed_size],state =[2,batch,]
        fw_out,bk_out = outputs
        output_rnn = tf.concat(outputs,axis=2)#shape = [batch,seq_len,self.embed_size*2]
        #SecondLstm
        rnn_cell = rnn.BasicLSTMCell(num_units=self.embed_size*2)
        if self.drop_keep_prob is not None:
            rnn_cell =rnn.DropoutWrapper(rnn_cell,output_keep_prob=self.dropout_keep_prob)
        _,final_state_c_h= tf.nn.dynamic_rnn(rnn_cell,output_rnn)
        final_state = final_state_c_h[1]
        #fc_layer
        out_put = tf.layers.dense(final_state,2*self.hidden_size,activation=tf.nn.tanh)
        return output

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





