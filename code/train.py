import os
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from config import Configurations
from data import EN2CNDataset, infinite_iter
from utils import tokens2sentence, computebleu, schedule_sampling
from model import build_model, save_model
from test import test


if __name__ == '__main__':
    model_name = 'gru_1_512'
    config = Configurations()
    config.store_model_path = f"../{model_name}_checkpoints/"  # 儲存模型的位置
    os.makedirs(config.store_model_path, exist_ok=True)

    """## 訓練流程
    - 先訓練，再檢驗
    """
    # 準備訓練資料
    train_dataset = EN2CNDataset(config, train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = infinite_iter(train_loader)
    # 準備檢驗資料
    val_dataset = EN2CNDataset(config, train=False)
    val_loader = DataLoader(val_dataset, batch_size=1)
    # 建構模型
    model, optimizer = build_model(config, train_dataset.src_vocab_size, train_dataset.tar_vocab_size)
    loss_function = CrossEntropyLoss(ignore_index=0)

    print(model)
    with open(f'{config.store_model_path}/log.txt', 'w') as f:
        f.write(str(model) + '\n')

    train_losses, val_losses, bleu_scores = [], [], []
    total_steps = 0
    while (total_steps < config.num_steps):
        # 訓練模型
        model.train()
        model.zero_grad()
        losses = []
        loss_sum = 0.0
        for step in range(config.summary_steps):
            sources, targets, src_mask, tar_mask = next(train_iter)  # sources targets[batch_size, max_output_len]

            src_seq_len = src_mask.sum(-1).detach().numpy()
            order = list(np.argsort(src_seq_len))[::-1]
            sources = sources[order]
            targets = targets[order]
            src_mask = src_mask[order]
            tar_mask = tar_mask[order]

            sources, targets = sources.to(config.device), targets.to(config.device)
            src_mask, tar_mask = src_mask.to(config.device), tar_mask.to(config.device)
            outputs, preds, loss, w_loss = model(sources, targets, src_mask, tar_mask, schedule_sampling())
            # outputs = [batch size, seq len, vocab size]
            # preds = [batch size, vocab size -1]
            # targets 的第一個 token 是 <BOS> 所以忽略
            outputs = outputs[:, 1:].reshape(-1, outputs.size(2))  # outputs = [batch size * (target len-1), vocab size]
            targets = targets[:, 1:].reshape(-1)  # targets = [batch size * (target len - 1)]
            # loss = loss_function(outputs, targets)

            optimizer.zero_grad()
            loss.mean().backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            loss_sum += loss.item()
            if (step + 1) % 10 == 0:
                loss_sum = loss_sum / 10
                print("train[{}] loss: {:.3f}, Perplexity: {:.3f}".format(total_steps + step + 1,
                                                                          loss_sum,
                                                                          np.exp(loss_sum)))
                with open(f'{config.store_model_path}/log.txt', 'a') as f:
                    f.write("train[{}] loss: {:.3f}, Perplexity: {:.3f}\n".format(total_steps + step + 1, loss_sum,
                                                                                  np.exp(loss_sum)))

                losses.append(loss_sum)
                loss_sum = 0.0
        train_losses += losses

        # 檢驗模型
        val_loss, bleu_score, result = test(model, val_loader, loss_function)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)
        total_steps += config.summary_steps
        print("val[{}] loss: {:.3f}, Perplexity: {:.3f}, blue score: {:.3f}".format(total_steps, val_loss,
                                                                                    np.exp(val_loss), bleu_score))
        with open(f'{config.store_model_path}/log.txt', 'a') as f:
            f.write("val[{}] loss: {:.3f}, Perplexity: {:.3f}, blue score: {:.3f}\n".format(total_steps, val_loss,
                                                                                            np.exp(val_loss),
                                                                                            bleu_score))

        # 儲存模型和結果
        save_model(model, optimizer, config.store_model_path, 'new')
        with open(f'{config.store_model_path}/test_{total_steps}.txt', 'w') as f:
            for line in result:
                res = '【待翻译】' + " ".join(line[0]) + '\n【Translation】' + " ".join(
                    line[1]) + '\n【GroundTruth】' + " ".join(line[2]) + '\n'
                print(res, file=f)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('次數')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.imsave(f'{config.store_model_path}/train loss.png', train_losses)

    """## 以圖表呈現 檢驗 的 loss 變化趨勢"""
    plt.figure()
    plt.plot(val_losses)
    plt.xlabel('次數')
    plt.ylabel('loss')
    plt.title('validation loss')
    plt.imsave(f'{config.store_model_path}/validation loss.png', val_losses)

    """## BLEU score"""
    plt.figure()
    plt.plot(bleu_scores)
    plt.xlabel('次數')
    plt.ylabel('BLEU score')
    plt.title('BLEU score')
    plt.imsave(f'{config.store_model_path}/BLEU score.png', bleu_scores)
