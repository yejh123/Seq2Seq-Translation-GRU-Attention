import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from model import build_model
from config import Configurations
from data import EN2CNDataset, infinite_iter
from utils import tokens2sentence, computebleu, schedule_sampling

"""## test
 test_epoch[2636] test loss: 4.986993104441314, bleu_score: 0.45033824057073923
"""


def test(model, dataloader, loss_function):
    model.eval()
    config = Configurations()
    loss_sum, bleu_score = 0.0, 0.0
    n = 0
    result = []
    for sources, targets, src_mask, tar_mask in dataloader:
        sources, targets = sources.to(config.device), targets.to(config.device)
        src_mask, tar_mask = src_mask.to(config.device), tar_mask.to(config.device)
        batch_size = sources.size(0)
        outputs, preds = model.inference(sources, targets, src_mask, tar_mask)
        # outputs = [batch size, sequence len, vocab size]  preds = [batch size, sequcence len -1]
        # targets 的第一個 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        sources = sources[:, 1:].reshape(-1)

        loss = loss_function(outputs, targets)
        loss_sum += loss.item()

        # 將預測結果轉為文字
        targets = targets.view(sources.size(0), -1)
        preds = tokens2sentence(preds, dataloader.dataset.int2word_tar)
        sources = tokens2sentence(sources, dataloader.dataset.int2word_src)
        targets = tokens2sentence(targets, dataloader.dataset.int2word_tar)
        for source, pred, target in zip(sources, preds, targets):
            result.append((source, pred, target))
        # 計算 Bleu Score
        bleu_score += computebleu(preds, targets)

        n += batch_size
        print("\r", f"test_epoch[{n}]", end=" ")
    return loss_sum / len(dataloader), bleu_score / n, result


def test_beam_search(config, model, dataloader, loss_function):
    model.eval()
    loss_sum, bleu_score = 0.0, 0.0
    n = 0
    result = []
    for sources, targets, src_mask, tar_mask in dataloader:
        sources, targets = sources.to(config.device), targets.to(config.device)
        src_mask, tar_mask = src_mask.to(config.device), tar_mask.to(config.device)
        batch_size = sources.size(0)
        preds = model.beam_search(sources, src_mask)  # [beam_size, seq_len]

        preds = tokens2sentence(preds[0][0], dataloader.dataset.int2word_tar, batch=False)
        sources = tokens2sentence(sources, dataloader.dataset.int2word_src)[1:]
        targets = tokens2sentence(targets, dataloader.dataset.int2word_tar)[1:]
        for source, pred, target in zip(sources, preds, targets):
            result.append((source, pred, target))
        # 計算 Bleu Score
        bleu_score += computebleu(preds, targets)

        n += batch_size
        print("\r", f"test_epoch[{n}]", end=" ")
    return loss_sum / len(dataloader), bleu_score / n, result


if __name__ == '__main__':
    model_name = 'gru_1_1024'
    config = Configurations()
    config.load_model = True
    config.load_model_path = f'../{model_name}_checkpoints/model_best.ckpt'
    print('config:\n', vars(config))

    # 準備測試資料
    test_dataset = EN2CNDataset(config, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1)
    config.SOS = test_dataset.SOS
    config.EOS = test_dataset.EOS
    config.PAD = test_dataset.PAD
    config.UNK = test_dataset.UNK

    # 建構模型
    model, optimizer = build_model(config, test_dataset.src_vocab_size, test_dataset.tar_vocab_size)
    print("Finish build model")
    loss_function = CrossEntropyLoss(ignore_index=0)
    model.eval()
    # 測試模型
    test_loss, bleu_score, result = test_beam_search(config, model, test_loader, loss_function)
    for line in result:
        res = '【待翻译】' + " ".join(line[0]) + '\n【Translation】' + " ".join(
            line[1]) + '\n【GroundTruth】' + " ".join(line[2]) + '\n'
        print(res)
    # 儲存結果
    # with open(f'{config.store_model_path}/{log_name}.txt', 'w') as f:
    #     for line in result:
    #         print(line, file=f)

    print(f'test loss: {test_loss}, bleu_score: {bleu_score}')

"""# 圖形化訓練過程

## 以圖表呈現 訓練 的 loss 變化趨勢
"""
