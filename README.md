# Seq2Seq-Translation-GRU-Attention
 - [P3n9W31/AttentionRNN](https://github.com/P3n9W31/AttentionRNN)
 - [完全图解RNN、RNN变体、Seq2Seq、Attention机制](https://zhuanlan.zhihu.com/p/28054589)
 - [Attention 扫盲：注意力机制及其 PyTorch 应用实现](https://zhuanlan.zhihu.com/p/88376673)
 
 
## Seq2Seq Model

``` pytorch
  Seq2Seq(
    (init_affine): Linear(in_features=2048, out_features=1024, bias=True)
    (encoder): Encoder(
      (embedding): Embedding(13290, 512)
      (rnn): GRU(512, 1024, batch_first=True, bidirectional=True)
      (enc_emb_dp): Dropout(p=0.4, inplace=False)
      (enc_hid_dp): Dropout(p=0.4, inplace=False)
    )
    (decoder): Decoder(
      (embedding): Embedding(11873, 512)
      (dec_emb_dp): Dropout(p=0.4, inplace=False)
      (attention): Attention(
        (h2s): Linear(in_features=1024, out_features=1000, bias=True)
        (s2s): Linear(in_features=2048, out_features=1000, bias=True)
        (a2o): Linear(in_features=1000, out_features=1, bias=True)
      )
      (gru1): GRUCell(512, 1024)
      (gru2): GRUCell(2048, 1024)
      (embedding2out): Linear(in_features=512, out_features=620, bias=True)
      (hidden2out): Linear(in_features=1024, out_features=620, bias=True)
      (c2o): Linear(in_features=2048, out_features=620, bias=True)
      (readout_dp): Dropout(p=0.4, inplace=False)
      (affine): Linear(in_features=620, out_features=11873, bias=True)
    )
  )
```
 
## Evaluation

The evaluation metric for Chinese-English we use is case-insensitive BLEU. We use the `muti-bleu.perl` script from [Moses](https://github.com/moses-smt/mosesdecoder) to compute the BLEU.

Loss and BLEU Score:

<p align="center">
<img src="https://github.com/yejh123/Seq2Seq-Translation-GRU-Attention/blob/main/gru_1_1024_checkpoints/train_loss.png" width="400">
<img src="https://github.com/yejh123/Seq2Seq-Translation-GRU-Attention/blob/main/gru_1_1024_checkpoints/bleu.png" width="400">
</p>

As the data I use is too simple, the results are **just a reference**.



## Results on Chinese-English translation
```
【待翻译】目前 台湾 已 有 超过 七成 的 上市 公司 向 银行 质押 自家 股票 , 其中 有 9 家 为 大 财团 。
【Translation】currently more than 70 percent of listed companies , including nine large financial groups , are operating with money they borrowed from banks with their stocks as collaterals .
【GroundTruth】currently more than 70 percent of listed companies , including nine large financial groups , are operating with money they borrowed from banks with their stocks as collaterals .

【待翻译】二 是 向 党政 领导 机关 、 行政 执法 人员 行贿 , 为 进行 走私 、 制假 等 寻求 “ 保护伞 ” 。
【Translation】2 . some people bribed party and government organs , administrative officers , and law enforcement personnel to seek " protection " for their smuggling and counterfeiting activities .
【GroundTruth】2 . some people bribed party and government organs , administrative officers , and law enforcement personnel to seek " protection " for their smuggling and counterfeiting activities .

【待翻译】面对 中外 记者 , 唐家璇 外长 勾画 的 新 世纪 中国 外交 走向 , 随着 电波 迅速 传 向 全球 。
【Translation】with this statement he made while talking to the chinese and foreign reporters , foreign affairs minister outlined china 's foreign policy for the new century .
【GroundTruth】with this statement he made while talking to the chinese and foreign reporters , foreign affairs minister outlined china 's foreign policy for the new century .

【待翻译】国会 共和党 内 的 反华 势力 , 极力 主张 台湾 先期 加入 世贸 组织 , 反对 给予 中国 pntr 地位 。
【Translation】the anti-china force in the congress republican party has vigorously advocated that taiwan should join the wto before china and has opposed the granting of pntr to china .
【GroundTruth】the anti-china force in the congress republican party has vigorously advocated that taiwan should join the wto before china and has opposed the granting of pntr to china .

【待翻译】认识 上 的 误区 使 他们 甚至 包括 一些 水利 干部 , 平时 的 工作 就 是 争取 资金 , 打井 取水 。
【Translation】because of their erroneous understanding , some cadres , including even water conservancy cadres , believe their work is to seek funds for drilling wells for water .
【GroundTruth】because of their erroneous understanding , some cadres , including even water conservancy cadres , believe their work is to seek funds for drilling wells for water .
```

Since sencentences tested above are part of training dataset, so it work better than in fact.





