import torch

from model import Encoder, Decoder, Seq2Seq

"""# utils
- 基本操作:
  - 儲存模型
  - 載入模型
  - 建構模型
  - 將一連串的數字還原回句子
  - 計算 BLEU score
  - 迭代 dataloader
"""


# 數字轉句子
def tokens2sentence(outputs, int2word, batch=True):
    sentences = []
    if batch:
        for tokens in outputs:
            sentence = []
            for token in tokens:
                word = int2word[int(token)]
                # if word == '<EOS>':
                if word == '<eos>':
                    break
                sentence.append(word)
            sentences.append(sentence)
    else:
        sentence = []
        for token in outputs:
            word = int2word[int(token)]
            # if word == '<EOS>':
            if word == '<eos>':
                break
            sentence.append(word)
        sentences.append(sentence)
    return sentences


"""## 計算 BLEU score"""

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def computebleu(sentences, targets):
    score = 0
    assert (len(sentences) == len(targets))

    def cut_token(sentence):
        tmp = []
        for token in sentence:
            # if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
            if token == '<unk>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))

    return score


"""## schedule_sampling"""


########
# TODO #
########

# 請在這裡直接 return 0 來取消 Teacher Forcing
# 請在這裡實作 schedule_sampling 的策略

def schedule_sampling():
    return 1
