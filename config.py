class MultilabelConfig:
    has_cuda = True
    is_training = True
    is_pretrain = True
    force_word2index = False
    embedding_path = "./word2vec/pretrain_emb.alltrain.256d.npy"
    test_path = './corpus/seg_test.txt'
    #    test_path = './data/test_preprocessed.txt'
    result_path = './results/test_result_task2.json'
    data_path = './corpus/seg_train.txt'
    #    data_path = './data/seg_full_shuffle_train.txt'
    #    data_path = './data/train_m_preprocessed.txt'
    model_path = './pickles/params.pkl'

    index2word_path = './idx2word.pkl'
    word2index_path = './word2idx.pkl'

    batch_size = 32  # 64 or larger if has cuda
    step = 10000 // batch_size  # 3000 // batch_size if has cuda
    num_workers = 1
    #    vocab_size = 241684
    #    vocab_size = 338209
    vocab_size = 0
    min_count = 5
    max_text_len = 2000
    embedding_size = 256
    num_class = 452
    #    num_class = 321
    learning_rate = 0.001
    if not is_pretrain:
        learning_rate2 = 0.001
    else:
        learning_rate2 = 0.0  # 0.0 if pre train emb
    lr_decay = 0.75
    begin_epoch = 2
    weight_decay = 0.0
    dropout_rate = 0.5
    epoch_num = 6
    epoch_step = max(1, epoch_num // 20)