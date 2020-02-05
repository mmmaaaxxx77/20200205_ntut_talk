import sys
import os

from pandas import DataFrame
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from ml_tools import (label_encode,
                      PredictDataPrecessForSingleSentence,
                      label_decode)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
logger = logging.getLogger(__name__)
stop_words = list(".,。，！!：:；;？?")


class InputProcessor:
    def __init__(self, max_workers=5):
        # data precess
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    @staticmethod
    def split_content(text):

        # lambda
        split_w = lambda t, s: [i for i in t.split(s) if len(i) != 0]

        ft = [text]
        for stop in stop_words:
            for i in range(0, len(ft)):
                # split len over max len (50)
                if len(ft[i]) <= (52 - 2):
                    ft[i] = [ft[i]]
                else:
                    ft[i] = split_w(ft[i], stop)
            if len(ft) > 0:
                ft = list(np.concatenate(ft))
        return ft

    def do(self, contents):

        # split content
        list_cons = list(
            self.pool.map(self.split_content, contents)
        )

        # convert to dataframe, [content, content_id]
        data = DataFrame([], columns=('text', 'id'))
        for i in range(0, len(list_cons)):
            item = list_cons[i]
            if item is None:
                data.append(['', i])
            else:
                for it in item:
                    data.loc[len(data)] = [
                        it if it is not None and len(it) != 0 else ''
                        ,
                        i
                    ]
        return data


class SentimentClassifier:
    def __init__(self):
        """Init."""
        self.processor = PredictDataPrecessForSingleSentence(bert_tokenizer=bert_tokenizer)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # model
        self.model = BertForSequenceClassification.from_pretrained(
            "model_path",
            num_labels=11
        )
        self.model.eval()
        self.model.to(self.device)

        self.input_processor = InputProcessor()

    def convert_predict_map(self, map):
        result = []

        do_pers = lambda x, _sum: [float(f"{(i / _sum):.5f}") for i in x]

        for i in range(0, len(map)):
            result.append(do_pers(map[i], sum(map[i])))
        return result

    def sum_labels(self, lis):
        return [lis[0], sum(lis[1:5]), sum(lis[6:])]  # 中立 正 負

    def predict(self, contents, return_all=False):
        data = self.input_processor.do(contents)
        print(data)

        seqs, seq_masks, seq_segments, labels = self.processor.get_input(
            dataset=data, max_seq_len=52)

        # to torch tensor
        t_seqs = torch.tensor(seqs, dtype=torch.long)
        t_seq_masks = torch.tensor(seq_masks, dtype=torch.long)
        t_seq_segments = torch.tensor(seq_segments, dtype=torch.long)
        t_labels = torch.tensor(labels, dtype=torch.long)

        pre_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)
        pre_dataloder = DataLoader(dataset=pre_data, batch_size=512)

        # predict
        ids_map = {}

        print(f"start predict: {len(pre_data)} docs")
        with torch.no_grad():
            for batch_data in pre_dataloder:
                batch_data = tuple(t.to(self.device) for t in batch_data)
                batch_seqs, batch_seq_masks, batch_seq_segments, batch_ids = batch_data
                logits = self.model(
                    batch_seqs, batch_seq_masks, batch_seq_segments)[0]
                logits = logits.softmax(dim=1)
                pred_result = logits.detach().numpy()
                _ids = batch_ids.detach().numpy()

                for i in range(0, len(_ids)):
                    _id = _ids[i]
                    _pred_labels = pred_result[i]
                    if not return_all:
                        _pred_labels = self.sum_labels(_pred_labels)
                    # group by id
                    if _id not in ids_map.keys():
                        ids_map[_id] = _pred_labels
                    else:
                        ids_map[_id] = (np.sum([ids_map[_id], _pred_labels], axis=0)).tolist()
        if return_all:
            return self.convert_predict_map(ids_map)

        return self.convert_predict_map(ids_map)


def demo():
    senti = SentimentClassifier()

    contents = [
        '現在警方加暴太盡，令到學生不怕死，只想攬住死',
        '台灣人要時時提醒自己，無差別攻擊的人和香港那邊的施暴者沒什麼不同,台灣人要時時提醒自己，無差別攻擊的人和香港那邊的施暴者沒什麼不同 台灣人要時時提醒自己，無差別攻擊的人和香港那邊的施暴者沒什麼不同,台灣人要時時提醒自己，無差別攻擊的人和香港那邊的施暴者沒什麼不同',
        '國民黨改選',
        '民進黨出了蔡英文總統',
        '滿意寶寶',
        '權值股熄火 台股開低走低下跌52點收11',
        '[MASK][MASK][MASK]宣布投入2020[MASK]',
        '感謝郭台銘先生',
        '記得要 #公開分享，FB抽獎系統才抓得到資料哦'
    ]

    result = senti.predict(contents)
    for i in range(0, len(contents)):
        print(contents[i])
        print(result[i])
        print(result[i].index(max(result[i])))
        print("--------")


if __name__ == '__main__':
    demo()
