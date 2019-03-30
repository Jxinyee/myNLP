import tensorflow as tf
class Fasttexttf:
    def __init__(self,config):
        self.config = config
        #add 占位符
        self.sentence = tf.placeholder(tf.int32,[None,self.config.sentence_len],name="sentence")
        self.label = tf.placeholder(tf.int32,[None,self.config.max_label_per_example],name="Labels")
        self.label_l199 =tf.placeholder(tf.float32,[None,self.config.label_size])

        # 设置变量
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.config.epoch_step, tf.add(self.config.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = self.config.decay_steps, self.config.decay_rate
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.init_weight()
        self.logits = self.inference()
        self.loss_val = self.loss()
        self.train_op = self.train()


    def init_weight(self):
        self.embedding = tf.get_variable("Embedding",[self.config.vocab_size,self.config.embeding_dim],initializer=self.config.initlizer)
        self.W = tf.get_variable("weight",[self.config.embeding_dim,self.config.label_size],initializer=self.config.initlizer)
        self.b = tf.get_variable("bias",[self.config.label_size])
    def inference(self):
        sentence_embeding = tf.nn.embedding_lookup(self.embedding,self.sentence)
        text_embed = tf.reduce_mean(sentence_embeding,axis=1)
        logits = tf.matmul(text_embed,self.W)+self.b
        return logits
    def loss(self,l2_lambda =0.001):
        labels_multi_hot = self.labels_l1999  # [batch_size,label_size]
        # sigmoid_cross_entropy_with_logits:Computes sigmoid cross entropy given `logits`.Measures the probability error in discrete classification tasks in which each class is independent and not mutually exclusive.  For instance, one could perform multilabel classification where a picture can contain both an elephant and a dog at the same time.
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_multi_hot,
                                                       logits=self.logits)  # labels:[batch_size,label_size];logits:[batch, label_size]
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))  # reduce_sum
        print("loss:", loss)

        # add regularization result in not converge
        self.l2_losses = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        print("l2_losses:", self.l2_losses)
        loss = loss + self.l2_losses
        return loss
    def train(self):
        learning_rate = tf.train.exponential_decay(self.config.learning_rate,global_step=self.global_step,decay_rate=self.config.decay_rate,staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op