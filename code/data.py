import torch
import torch.utils.data as data

import numpy as np
import re
import sys
import os
import random
import json
import pickle


def load_vocab(path):
    f = open(path, 'rb')
    vocab = pickle.load(f)
    f.close()
    return vocab


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines


"""# 資料結構

## 定義資料的轉換
- 將不同長度的答案拓展到相同長度，以便訓練模型
"""


class LabelTransform(object):
    def __init__(self, size, pad):
        self.size = size
        self.pad = pad

    def __call__(self, label):
        label = np.pad(label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
        return label


class EN2CNDataset(data.Dataset):
    def __init__(self, config, train=True):
        self.config = config
        self.src_vocab = config.src_vocab
        self.tar_vocab = config.tar_vocab
        if train:
            self.src_path = config.src_train_path
            self.tar_path = config.tar_train_path
        else:
            self.src_path = config.src_val_path
            self.tar_path = config.tar_val_path

        self.word2int_src, self.int2word_src = self.get_dictionary(self.src_vocab)
        self.word2int_tar, self.int2word_tar = self.get_dictionary(self.tar_vocab)
        self.SOS = self.word2int_src['<sos>']
        self.EOS = self.word2int_src['<eos>']
        self.UNK = self.word2int_src['<unk>']
        self.PAD = self.word2int_src['<pad>']
        self.src_vocab_size = len(self.word2int_src)
        self.tar_vocab_size = len(self.word2int_tar)
        # self.transform = LabelTransform(config.max_output_len, self.word2int_en['<PAD>'])
        self.transform = LabelTransform(config.max_output_len, self.word2int_src['<pad>'])

        # 載入資料
        self.src_data = load_data(self.src_path)
        self.tar_data = load_data(self.tar_path)
        self.src_mask = []
        self.tar_mask = []
        assert len(self.src_data) == len(self.tar_data)
        print(f'dataset size: {len(self.src_data)}')
        self.init_data()

    def get_dictionary(self, file):
        # 載入字典
        # with open(os.path.join(self.root, f'word2int_{language}.json'), "r", encoding="utf-8") as f:
        #     word2int = json.load(f, encoding="utf-8")
        # with open(os.path.join(self.root, f'int2word_{language}.json'), "r", encoding="utf-8") as f:
        #     int2word = json.load(f, encoding="utf-8")
        # return word2int, int2word

        word2int = load_vocab(file)
        int2word = {}
        for k in word2int:
            int2word[word2int[k]] = k
        return word2int, int2word

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tar_data[idx], self.src_mask[idx], self.tar_mask[idx]

    def init_data(self):
        # 在開頭添加 <BOS>，在結尾添加 <EOS> ，不在字典的 subword (詞) 用 <UNK> 取代
        # 將句子拆解為 subword 並轉為整數
        # sentence = re.split(' ', sentences[0])
        # sentence = list(filter(None, sentence))
        src_list = []
        src_mask_list = []
        for i in range(len(self.src_data)):
            src = [self.SOS]
            line = self.src_data[i].strip().split(' ')
            for word in line:
                src.append(self.word2int_src.get(word, self.UNK))
                if len(src) == self.config.max_output_len - 1:
                    break
            src.append(self.EOS)
            src = np.asarray(src)
            # 用 <PAD> 將句子補到相同長度
            src = self.transform(src)
            src = torch.LongTensor(src)
            src_list.append(src)
            src_mask = src.ne(self.PAD).float()
            src_mask_list.append(src_mask)
        self.src_data = src_list
        self.src_mask = src_mask_list

        tar_list = []
        tar_mask_list = []
        for i in range(len(self.tar_data)):
            tar = [self.SOS]
            line = self.tar_data[i].strip().split(' ')
            for word in line:
                tar.append(self.word2int_tar.get(word, self.UNK))
                if len(tar) == self.config.max_output_len - 1:
                    break
            tar.append(self.EOS)
            tar = np.asarray(tar)
            # 用 <PAD> 將句子補到相同長度
            tar = self.transform(tar)
            tar = torch.LongTensor(tar)
            tar_list.append(tar)
            tar_mask = tar.ne(self.PAD).float()
            tar_mask_list.append(tar_mask)
        self.tar_data = tar_list
        self.tar_mask = tar_mask_list


def infinite_iter(data_loader):
    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)


if __name__ == '__main__':
    from config import Configurations

    config = Configurations()
    train_dataset = EN2CNDataset(config)

    print('suc')
