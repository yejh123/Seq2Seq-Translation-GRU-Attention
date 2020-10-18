import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
from beam import Beam

"""# 模型架構
## Encoder
- seq2seq模型的編碼器為RNN。 對於每個輸入，，**Encoder** 會輸出**一個向量**和**一個隱藏狀態(hidden state)**，
  並將隱藏狀態用於下一個輸入，換句話說，**Encoder** 會逐步讀取輸入序列，並輸出單個矢量（最終隱藏狀態）
- 參數:
  - en_vocab_size 是英文字典的大小，也就是英文的 subword 的個數
  - emb_dim 是 embedding 的維度，主要將 one-hot vector 的單詞向量壓縮到指定的維度，主要是為了降維和濃縮資訊的功用，
  可以使用預先訓練好的 word embedding，如 Glove 和 word2vector
  - hid_dim 是 RNN 輸出和隱藏狀態的維度
  - n_layers 是 RNN 要疊多少層
  - dropout 是決定有多少的機率會將某個節點變為 0，主要是為了防止 overfitting ，一般來說是在訓練時使用，測試時則不使用
- Encoder 的輸入和輸出:
  - 輸入: 
    - 英文的整數序列 e.g. 1, 28, 29, 205, 2
  - 輸出: 
    - outputs: 最上層 RNN 全部的輸出，可以用 Attention 再進行處理
    - hidden: 每層最後的隱藏狀態，將傳遞到 Decoder 進行解碼

## Decoder
- **Decoder** 是另一個 RNN，在最簡單的 seq2seq decoder 中，僅使用 **Encoder** 每一層最後的隱藏狀態來進行解碼，
  而這最後的隱藏狀態有時被稱為 “content vector”，因為可以想像它對整個前文序列進行編碼， 
  此 “content vector” 用作 **Decoder** 的**初始**隱藏狀態， 而 **Encoder** 的輸出通常用於 Attention Mechanism
- 參數
  - cn_vocab_size 是中文字典的大小，也就是中文的 word 的個數
  - emb_dim 是 embedding 的維度，是用來將 one-hot vector 的單詞向量壓縮到指定的維度，主要是為了降維和濃縮資訊的功用，
    可以使用預先訓練好的 word embedding，如 Glove 和 word2vector
  - hid_dim 是 RNN 輸出和隱藏狀態的維度
  - output_dim 是最終輸出的維度，一般來說是將 hid_dim 轉到 one-hot vector 的單詞向量
  - n_layers 是 RNN 要疊多少層
  - dropout 是決定有多少的機率會將某個節點變為0，主要是為了防止 overfitting ，一般來說是在訓練時使用，測試時則不用
  - isatt 是來決定是否使用 Attention Mechanism
- Decoder 的輸入和輸出:
- 輸入:
  - 前一次解碼出來的單詞的整數表示
- 輸出:
  - hidden: 根據輸入和前一次的隱藏狀態，現在的隱藏狀態更新的結果
  - output: 每個字有多少機率是這次解碼的結果
    
## Attention
- 當輸入過長，或是單獨靠 “content vector” 無法取得整個輸入的意思時，用 Attention Mechanism 來提供 **Decoder** 更多的資訊
- 主要是根據現在 **Decoder hidden state** ，去計算在 **Encoder outputs** 中，那些與其有較高的關係，
  根據關系的數值來決定該傳給 **Decoder** 那些額外資訊 
- 常見 Attention 的實作是用 Neural Network / Dot Product 來算 **Decoder hidden state** 和 **Encoder outputs** 
  之間的關係，再對所有算出來的數值做 **softmax** ，最後根據過完 **softmax** 的值對 **Encoder outputs** 做 **weight sum**
- TODO:
實作 Attention Mechanism

## Seq2Seq
- 由 **Encoder** 和 **Decoder** 組成
- 接收輸入並傳給 **Encoder** 
- 將 **Encoder** 的輸出傳給 **Decoder**
- 不斷地將 **Decoder** 的輸出傳回 **Decoder** ，進行解碼  
- 當解碼完成後，將 **Decoder** 的輸出傳回
"""


class Encoder(nn.Module):
    def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(en_vocab_size, emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True, bidirectional=True)
        self.enc_emb_dp = nn.Dropout(dropout)
        self.enc_hid_dp = nn.Dropout(dropout)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        h0 = weight.new_zeros(2 * self.n_layers, batch_size, self.hid_dim)
        return h0

    def forward(self, input, src_mask):
        """
        :param input: [N, seq_len]
        :param src_mask: [N, seq_len]
        """
        emb = self.enc_emb_dp(self.embedding(input))  # embedding = [N, seq_len, emb dim]
        length = src_mask.sum(1).tolist()
        total_length = src_mask.size(1)
        emb = nn.utils.rnn.pack_padded_sequence(emb, length, batch_first=True)

        hidden = self.init_hidden(input.size(0))  # [2*n_layers, N, hid_dim]
        outputs, hidden = self.rnn(emb, hidden)
        # outputs = [N, seq_len, hid_dim * directions]
        # hidden = [n_layers * directions, N, hid_dim]
        # outputs 是最上層RNN的輸出

        outputs = torch.nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=total_length
        )[0]
        outputs = self.enc_hid_dp(outputs)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, tar_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt, nreadout=620):
        super().__init__()
        self.tar_vocab_size = tar_vocab_size
        self.n_layers = n_layers
        self.isatt = isatt
        self.hid_dim = hid_dim  # 注意这里的维度变化 hid_dim*2
        self.enc_ncontext = hid_dim * 2
        # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
        # e.g. Attention 接在輸入後面會使得維度變化，所以輸入維度改為
        # self.input_dim = emb_dim
        # self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim

        self.embedding = nn.Embedding(tar_vocab_size, emb_dim)
        self.dec_emb_dp = nn.Dropout(dropout)
        self.attention = Attention(self.hid_dim, 2 * self.hid_dim, 1000)

        # self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout=dropout, batch_first=True)
        # self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
        # self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
        # self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)

        self.gru1 = nn.GRUCell(emb_dim, self.hid_dim)  # emb_dim -> dec_hid_dim
        self.gru2 = nn.GRUCell(self.enc_ncontext, self.hid_dim)  # 2*enc_hid_dim -> dec_hid_dim
        self.attention = Attention(self.hid_dim, self.enc_ncontext, 1000)
        self.embedding2out = nn.Linear(emb_dim, nreadout)
        self.hidden2out = nn.Linear(self.hid_dim, nreadout)
        self.c2o = nn.Linear(self.enc_ncontext, nreadout)
        self.readout_dp = nn.Dropout(dropout)
        self.affine = nn.Linear(nreadout, self.tar_vocab_size)  # nreadout -> dec_num_token

    def forward(self, input, hidden, enc_context, src_mask_bool):
        """
        P.S. dec_hid_dim = enc_hid_dim * 2
        :param input: [N, ]
        :param hidden: [n_layers, N, dec_hid_dim]
        :param enc_context: [N, seq_len, dec_hid_dim]
        Decoder 只會是單向，所以 directions=1
        """
        input = input.view(-1, 1)
        emb = self.dec_emb_dp(self.embedding(input))  # dec_input = [N, 1, emb_dim]

        # if self.isatt:
        #     attn = self.attention(hidden, enc_context, src_mask_bool).unsqueeze(1)  # attn = [N, 1, dec_hid_dim]
        #     dec_input = torch.cat([dec_input, attn], -1)
        #
        # output, hidden = self.rnn(emb, hidden)
        # # output = [N, 1, dec_hid_dim]
        # # hidden = [n_layers, N, dec_hid_dim]
        #
        # # 將 RNN 的輸出轉為每個詞出現的機率
        # output = self.embedding2vocab1(output)
        # # output = [N, hid_dim*4]
        # output = self.embedding2vocab2(output)
        # # output = [N, hid_dim*8]
        # prediction = self.embedding2vocab3(output)
        # # prediction = [N, vocab size]
        # return prediction, hidden

        emb = emb.squeeze(1)  # [N, emb_dim]
        hidden = hidden.view(len(emb), -1)  # [N, dec_hid_dim]
        hidden = self.gru1(emb, hidden)  # [N, dec_hid_dim]
        attn_enc = self.attention(hidden, enc_context, src_mask_bool)  # [N, 2*enc_hid_dim]
        hidden = self.gru2(attn_enc, hidden)  # [N, dec_hid_dim]
        output = torch.tanh(self.embedding2out(emb) + self.hidden2out(hidden) + self.c2o(attn_enc))  # [N, nreadout]
        output = self.readout_dp(output)
        prediction = self.affine(output)
        return prediction, hidden


class Attention(nn.Module):
    def __init__(self, hid_dim, n_context, n_att):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim
        self.h2s = nn.Linear(hid_dim, n_att)  # enc_hid_imd -> n_att(1000)
        self.s2s = nn.Linear(n_context, n_att)  # 2*enc_hid_imd -> n_att(1000)
        self.a2o = nn.Linear(n_att, 1)  # n_att(1000) -> 1

    def forward(self, hidden, enc_context, src_mask_bool):
        """
        :param hidden: [n_layers, N, dec_hid_dim] or [N, dec_hid_dim] when n_layers=1
        :param src_mask_bool: [N, enc_max_len]
        :param enc_context: [N, seq_len, 2*enc_hid_dim]
        一般來說是取最後一層的 decoder hidden state 來做 attention
        """
        shape = enc_context.size()  # shape[N, seq_len, dec_hid_dim]
        # hidden = hidden[-1]  # [N, dec_hid_dim]

        attn_h = self.s2s(enc_context.reshape(-1, shape[2]))  # [N*seq_len, n_att]
        attn_h = attn_h.view(shape[0], shape[1], -1)  # [N, seq_len, n_att]
        attn_h += self.h2s(hidden).unsqueeze(1).expand_as(attn_h)  # [N, seq_len, n_att]
        logit = self.a2o(torch.tanh(attn_h)).squeeze(-1)  # [N, seq_len]
        if src_mask_bool.any():
            logit.data.masked_fill_(~src_mask_bool, -float("inf"))
        softmax = F.softmax(logit, dim=1)  # [N, seq_len]
        output = torch.matmul(softmax.unsqueeze(1), enc_context).squeeze(1)
        # [N, 1, seq_len] * [N, seq_len, 2*enc_hid_dim]
        return output  # [N, 2*enc_hid_dim]


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, config):
        super().__init__()
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"
        self.init_affine = nn.Linear(2 * config.hid_dim, config.hid_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.device = config.device

    def forward(self, input, target, src_mask, tar_mask, teacher_forcing_ratio):
        """
        :param input: [N, seq_len]
        :param target: [N, seq_len]
        :param tar_mask: [N, seq_len]
        :param src_mask: [N, seq_len]
        :param teacher_forcing_ratio: 是有多少機率使用正確答案來訓練
        Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        encoder_outputs 主要是使用在 Attention
        因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        """
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.tar_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        # outputs [batch_size, seq_len, cn_vocab_size]
        enc_context, enc_hidden = self.encoder(input, src_mask)
        # enc_context = [N, seq_len, hid_dim * directions]  enc_hidden = [n_layers * directions, N, hid_dim]

        avg_enc_context = enc_context.sum(1)  # [N, 2*enc_hid_dim]
        enc_context_len = src_mask.sum(1).unsqueeze(-1).expand_as(avg_enc_context)  # [N, 2*enc_hid_dim]
        avg_enc_context = avg_enc_context / enc_context_len  # [N, 2*enc_hid_dim]
        hidden = torch.tanh(self.init_affine(avg_enc_context))  # [N, dec_hid_dim]

        # hidden = [n_layers * directions, N, hid_dim]  -->
        # [n_layers, directions, N, hid_dim] -->
        # [n_layers, N, hid_dim * directions]
        # hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        # hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)

        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        loss = 0.0
        src_mask_bool = src_mask.bool()  # [N, max_len]
        for t in range(1, target_len):
            dec_prediction, hidden = self.decoder(input, hidden, enc_context, src_mask_bool)
            # dec_prediction = [N, 1, vocab size]
            # hidden = [n_layers, N, dec_hid_dim]   dec_hid_dim = 2 * enc_hid_dim
            dec_prediction = dec_prediction.squeeze(1)
            outputs[:, t] = dec_prediction
            top1 = dec_prediction.argmax(1)  # top1 = [N]
            preds.append(top1.unsqueeze(1))
            # 決定是否用正確答案來做訓練，如果是 teacher force 則用正解訓練，反之用自己預測的單詞做預測
            teacher_force = random.random() <= teacher_forcing_ratio
            input = target[:, t] if teacher_force and t < target_len else top1
            # 每次都添加loss
            loss += (F.cross_entropy(dec_prediction, target[:, t], reduction="none") * tar_mask[:, t])

        preds = torch.cat(preds, 1)  # [N, vocab size -1]
        w_loss = loss.sum() / tar_mask[:, 1:].sum()
        loss = loss.mean()
        return outputs, preds, loss.unsqueeze(0), w_loss.unsqueeze(0)

    def inference(self, input, target, src_mask, tar_mask=None):
        """
        :param input: [N, seq_len]
        :param target: [N, seq_len]
        :param src_mask: [N, seq_len]
        :param tar_mask: [N, seq_len]
        :return:
        """
        batch_size = input.shape[0]
        input_len = input.shape[1]  # 取得最大字數
        vocab_size = self.decoder.tar_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        enc_context, enc_hidden = self.encoder(input, src_mask)
        # enc_context = [N, seq_len, hid_dim * directions]  enc_hidden = [n_layers * directions, N, hid_dim]
        # hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        # hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)

        avg_enc_context = enc_context.sum(1)  # [N, 2*enc_hid_dim]
        enc_context_len = src_mask.sum(1).unsqueeze(-1).expand_as(avg_enc_context)  # [N, 2*enc_hid_dim]
        avg_enc_context = avg_enc_context / enc_context_len  # [N, 2*enc_hid_dim]
        hidden = torch.tanh(self.init_affine(avg_enc_context))  # [N, dec_hid_dim]

        src_mask_bool = src_mask.bool()  # [N, max_len]
        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, input_len):
            output, hidden = self.decoder(input, hidden, enc_context, src_mask_bool)
            output = output.squeeze(1)
            # 將預測結果存起來
            outputs[:, t] = output
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            input = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

    def beam_search(self, src, src_mask, beam_size=10, normalize=False, max_len=None, min_len=None):
        """
        :param src:
        :param src_mask:
        :param beam_size:
        :param normalize:
        :param max_len:
        :param min_len:
        :return:
        """
        batch_size = len(src)
        max_len = src.size(1) * 3 if max_len is None else max_len
        min_len = src.size(1) / 2 if min_len is None else min_len

        enc_context, _ = self.encoder(src, src_mask)
        enc_context = enc_context.contiguous()

        avg_enc_context = enc_context.sum(1)  # [N, 2*enc_hid_dim]
        enc_context_len = src_mask.sum(1).unsqueeze(-1).expand_as(avg_enc_context)  # [N, 2*enc_hid_dim]
        avg_enc_context = avg_enc_context / enc_context_len  # [N, 2*enc_hid_dim]
        hidden = torch.tanh(self.init_affine(avg_enc_context))  # [N, dec_hid_dim]

        # Beam search init
        prev_beam = Beam(beam_size)
        prev_beam.candidates = [[self.config.SOS]]
        prev_beam.scores = [0]
        f_done = lambda x: x[-1] == self.config.EOS

        valid_size = beam_size
        hyp_list = []
        attn_mask = src_mask.bool()
        for k in range(max_len):
            candidates = prev_beam.candidates
            input = src.new_tensor([cand[-1] for cand in candidates])
            dec_prediction, hidden = self.decoder(input, hidden, enc_context, attn_mask)
            log_prob = F.log_softmax(dec_prediction, dim=1)
            if k < min_len:
                log_prob[:, self.config.EOS] = -float("inf")
            if k == max_len - 1:
                eos_prob = log_prob[:, self.config.EOS].clone()
                log_prob[:, :] = -float("inf")
                log_prob[:, self.config.EOS] = eos_prob
            next_beam = Beam(valid_size)
            done_list, remain_list = next_beam.step(-log_prob, prev_beam, f_done)
            hyp_list.extend(done_list)
            valid_size -= len(done_list)

            if valid_size == 0:
                break

            beam_remain_ix = src.new_tensor(remain_list)
            enc_context = enc_context.index_select(0, beam_remain_ix)
            attn_mask = attn_mask.index_select(0, beam_remain_ix)
            hidden = hidden.index_select(0, beam_remain_ix)
            prev_beam = next_beam
        score_list = [hyp[1] for hyp in hyp_list]
        hyp_list = [
            hyp[0][1: hyp[0].index(self.config.EOS)]
            if self.config.EOS in hyp[0]
            else hyp[0][1:]
            for hyp in hyp_list
        ]
        if normalize:
            for k, (hyp, score) in enumerate(zip(hyp_list, score_list)):
                if len(hyp) > 0:
                    score_list[k] = score_list[k] / len(hyp)
        score = hidden.new_tensor(score_list)
        sort_score, sort_ix = torch.sort(score)
        output = []
        for ix in sort_ix.tolist():
            output.append((hyp_list[ix], score[ix].item()))
        return output


def build_model(config, en_vocab_size, tar_vocab_size):
    # 建構模型
    encoder = Encoder(en_vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout)
    decoder = Decoder(tar_vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout, config.attention)
    model = Seq2Seq(encoder, decoder, config)

    # 建構 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(optimizer)
    if config.load_model:
        model = load_model(model, config.load_model_path)
    model = model.to(config.device)

    return model, optimizer


def save_model(model, optimizer, store_model_path, step):
    torch.save(model.state_dict(), f'{store_model_path}/model_{step}.ckpt')
    return


def load_model(model, load_model_path):
    print(f'Load model from {load_model_path}')
    model.load_state_dict(torch.load(f'{load_model_path}'))
    return model
