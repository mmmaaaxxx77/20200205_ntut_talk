import os
import sys

from transformers import BertForSequenceClassification, AdamW, WarmupLinearSchedule
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
from tqdm import tqdm_notebook, trange
import matplotlib.pyplot as plt
import argparse

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from ml_tools import (label_encode, DataGenerator, DataPrecessForSingleSentence)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)


def train(args):
    # gen data
    dg = DataGenerator()
    # gen data
    data = dg.gen_df_data(args.data_dir, max_seq_len=args.max_seq_len)

    # data preprocessor
    processor = DataPrecessForSingleSentence(bert_tokenizer=bert_tokenizer)

    # data preprocess
    seqs, seq_masks, seq_segments, labels = processor.get_input(
        dataset=data, max_seq_len=int(52))

    # create model
    model = BertForSequenceClassification.from_pretrained(
        f'{args.model}', num_labels=11)

    # 轉換為torch tensor
    t_seqs = torch.tensor(seqs, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype=torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype=torch.long)
    t_labels = torch.tensor(labels, dtype=torch.long)

    train_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)
    train_dataloder = DataLoader(dataset=train_data, shuffle=True, batch_size=args.batch_size)

    model.train()

    # 待優化的參數
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {
            'params':
                [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
                0.01
        },
        {
            'params':
                [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=2e-05,
                      correct_bias=False)
    num_total_steps = 1000
    num_warmup_steps = 100
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)

    device = args.device
    model.to(device)

    # 存储每一个batch的loss
    loss_collect = []
    for i in trange(args.epoch_size, desc='Epoch'):
        for step, batch_data in enumerate(
                tqdm_notebook(train_dataloder, desc='Iteration')):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data

            logits = model(
                batch_seqs, batch_seq_masks, batch_seq_segments, labels=batch_labels)[0]
            logits.backward()
            loss_collect.append(logits.item())
            print("\r loss:%f" % logits, end='')
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # show chart
    if args.show_plt_loss:
        plt.figure(figsize=(12, 8))
        plt.plot(range(len(loss_collect)), loss_collect, 'g.')
        plt.grid(True)
        plt.savefig(f"{args.output_dir}/loss.png")
        plt.show()

    model.save_pretrained(args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model", default='bert-base-multilingual-cased', type=str, required=False,
                        help="The pretrain model name or path")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--epoch_size", default=20, type=int, help="Epoch size.")
    parser.add_argument("--max_seq_len", default=52, type=int, help="max_seq_len size.")
    parser.add_argument("--show_plt_loss", action='store_true', help="Show loss chat when done training.")
    args = parser.parse_args()

    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Device: {args.device}")

    print(f"show_plt_loss: {args.show_plt_loss}")

    print(f"data_dir: {args.data_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"model: {args.model}")
    print(f"batch_size: {args.batch_size}")
    print(f"epoch_size: {args.epoch_size}")
    print(f"max_seq_len: {args.max_seq_len}")

    train(args)


if __name__ == '__main__':
    main()

    # PYTHONIOENCODING=utf-8 python train.py --data_dir news.csv --output_dir /models/ --show_plt_loss --epoch_size 10 --batch_size 64
