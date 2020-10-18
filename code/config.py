"""# Config
- 實驗的參數設定表
"""
import os
import torch


class Configurations(object):
    def __init__(self):
        self.device = torch.device('cuda:0')

        # 旧数据集 英 -> 中
        # self.train_file_path = '../cmn-eng/training_small.txt'
        # self.test_file_path = '../cmn-eng/testing_small.txt'
        # self.val_file_path = '../cmn-eng/validation_small.txt'

        # 新数据集 中 -> 英
        self.src_vocab = '../data/cn.voc.pkl'
        self.tar_vocab = '../data/en.voc.pkl'
        self.src_train_path = '../data/cn.txt'
        self.tar_train_path = '../data/en.txt'
        self.src_val_path = '../data/cn.test.txt'
        self.tar_val_path = '../data/en.test.txt'

        # 特殊标识
        self.SOS = 0
        self.EOS = 1
        self.PAD = 2
        self.UNK = 3

        self.batch_size = 64
        self.emb_dim = 512
        self.hid_dim = 1024
        self.n_layers = 1
        self.dropout = 0.4
        self.learning_rate = 0.0005
        self.max_output_len = 35  # 最後輸出句子的最大長度
        self.num_steps = 5000  # 總訓練次數
        self.store_steps = 200  # 訓練多少次後須儲存模型
        self.summary_steps = 200  # 訓練多少次後須檢驗是否有overfitting
        self.load_model = False  # 是否需載入模型
        self.store_model_path = "../ckpt_v1.0"  # 儲存模型的位置
        self.load_model_path = None  # 載入模型的位置 e.g. "./ckpt_v0.2/model_{step}"
        self.data_path = "../cmn-eng"  # 資料存放的位置
        self.attention = True  # 是否使用 Attention Mechanism
