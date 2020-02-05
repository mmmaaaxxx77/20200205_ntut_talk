import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

# 負值為負面情緒程度, 0為中立, 正值為正面情緒程度
label_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    -5: 6,
    -4: 7,
    -3: 8,
    -2: 9,
    -1: 10,
}


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()
        print('time cost: ', self.end_time - self.start_time, 's')


timer = Timer()


def label_encode(label):
    """
    encode label
    input:
        label       : label need to encode
    output:
        label       : encode label
    """
    return label_map[label]


def label_decode(label):
    """
    decode label
    input:
        label       : label need to decode
    output:
        label       : decode label
    """
    return list(label_map.keys())[list(label_map.values()).index(label)]


class DataGenerator:

    def __init__(self, max_workers=5):
        self.max_workers = max_workers
        self.pool = ThreadPoolExecutor(max_workers=self.max_workers)

    def gen_df_data(self, filepath, max_seq_len=32):
        """
        gen train data
        input:
            filepath             : csv path
            ================
            | text | label |
            ================
            labels: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        output:
            dataset (DataFrame)  : dataset
        """
        df = pd.read_csv(filepath)
        df['label'].replace(-5, label_encode(-5), inplace=True)
        df['label'].replace(-4, label_encode(-4), inplace=True)
        df['label'].replace(-3, label_encode(-3), inplace=True)
        df['label'].replace(-2, label_encode(-2), inplace=True)
        df['label'].replace(-1, label_encode(-1), inplace=True)
        df.columns = ['text', 'label']

        # drop null
        df.dropna(inplace=True)

        # check type
        df['text'] = df['text'].astype('str')
        df['label'] = df['label'].astype('int')

        # filter text len if range under 30
        mask = (df['text'].str.len() < (max_seq_len - 2))
        df = df.loc[mask]

        # clean
        print(f"do replace...")
        timer.start()
        timer.end()
        df.replace([np.inf, -np.inf], np.nan)
        df.dropna()

        # reset dataframe index
        df = df.reset_index(drop=True)

        print(df)

        return df


class PredictDataPrecessForSingleSentence(object):
    """
    文本處理
    """

    def __init__(self, bert_tokenizer, max_workers=5):
        self.bert_tokenizer = bert_tokenizer
        self.max_workers = max_workers
        self.pool = ThreadPoolExecutor(max_workers=self.max_workers)

    def get_input(self,
                  dataset,
                  max_seq_len=32):
        """
        get tokenizer data

        input:
            dataset     : pandas的dataframe格式，包含兩列，第一列為文本，第二列為標簽。標簽取值為{0,1}，其中0表示負樣本，1代表正樣本。
            max_seq_len : 目標序列長度，該值需要預先對文本長度進行分別得到，可以設置為小於等於512（BERT的最長文本序列長度為512）的整數。

        output:
            seq         : 在入參seq的頭尾分別拼接了'CLS'與'SEP'符號，如果長度仍小於max_seq_len，則使用0在尾部進行了填充。
            seq_mask    : 只包含0、1且長度等於seq的序列，用於表徵seq中的符號是否是有意義的，如果seq序列對應位上為填充符號，
                          那麼取值為1，否則為0。
            seq_segment : shape等於seq，因為是單句，所以取值都為0。
            labels      : 標簽取值為{0,1}，其中0表示負樣本，1代表正樣本。


        """
        sentences = dataset.iloc[:, 0].tolist()
        ids = dataset.iloc[:, 1].tolist()
        # 切詞
        tokens_seq = list(
            self.pool.map(self.bert_tokenizer.tokenize, sentences))
        # 獲取定長序列及其mask
        result = list(
            self.pool.map(self.trunate_and_pad, tokens_seq,
                          [max_seq_len] * len(tokens_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return seqs, seq_masks, seq_segments, ids

    def trunate_and_pad(self,
                        seq,
                        max_seq_len: int):
        """
        1. 因為本類處理的是單句序列，按照BERT中的序列處理方式，需要在輸入序列頭尾分別拼接特殊字元'CLS'與'SEP'，
           因此不包含兩個特殊字元的序列長度應該小於等於max_seq_len-2，如果序列長度大於該值需要那麼進行截斷。
        2. 對輸入的序列 最終形成['CLS',seq,'SEP']的序列，該序列的長度如果小於max_seq_len，那麼使用0進行填充。

        input:
            seq         : 輸入序列，在本處其為單個句子。
            max_seq_len : 拼接'CLS'與'SEP'這兩個特殊字元後的序列長度

        output:
            seq         : 在入參seq的頭尾分別拼接了'CLS'與'SEP'符號，如果長度仍小於max_seq_len，則使用0在尾部進行了填充。
            seq_mask    : 只包含0、1且長度等於seq的序列，用於表徵seq中的符號是否是有意義的，如果seq序列對應位上為填充符號，
                          那麼取值為1，否則為0。
            seq_segment : shape等於seq，因為是單句，所以取值都為0。

        """
        # 對超長序列進行截斷
        if len(seq) > (max_seq_len - 2):
            seq = seq[0:(max_seq_len - 2)]
        # 分別在首尾拼接特殊符號
        seq = ['[CLS]'] + seq + ['[SEP]']
        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根據max_seq_len與seq的長度產生填充序列
        padding = [0] * (max_seq_len - len(seq))
        # 創建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 創建seq_segment
        seq_segment = [0] * len(seq) + padding
        # 對seq拼接填充序列
        seq += padding
        return seq, seq_mask, seq_segment


class DataPrecessForSingleSentence(object):
    """
    文本處理
    """

    def __init__(self, bert_tokenizer, max_workers=5):
        self.bert_tokenizer = bert_tokenizer
        self.max_workers = max_workers
        self.pool = ThreadPoolExecutor(max_workers=self.max_workers)

    def get_input(self,
                  dataset,
                  max_seq_len=32):
        """
        get tokenizer data

        input:
            dataset     : pandas的dataframe格式，包含兩列，第一列為文本，第二列為標簽。標簽取值為{0,1}，其中0表示負樣本，1代表正樣本。
            max_seq_len : 目標序列長度，該值需要預先對文本長度進行分別得到，可以設置為小於等於512（BERT的最長文本序列長度為512）的整數。

        output:
            seq         : 在入參seq的頭尾分別拼接了'CLS'與'SEP'符號，如果長度仍小於max_seq_len，則使用0在尾部進行了填充。
            seq_mask    : 只包含0、1且長度等於seq的序列，用於表徵seq中的符號是否是有意義的，如果seq序列對應位上為填充符號，
                          那麼取值為1，否則為0。
            seq_segment : shape等於seq，因為是單句，所以取值都為0。
            labels      : 標簽取值為{0,1}，其中0表示負樣本，1代表正樣本。


        """
        timer = Timer()
        sentences = dataset.iloc[:, 0].tolist()
        labels = dataset.iloc[:, 1].tolist()
        # 切詞
        print(f"do tokenize...")
        timer.start()
        tokens_seq = list(
            self.pool.map(self.bert_tokenizer.tokenize, sentences))
        timer.end()
        # 獲取定長序列及其mask
        print(f"do trunate_and_pad...")
        timer.start()
        result = list(
            self.pool.map(self.trunate_and_pad, tokens_seq,
                          [max_seq_len] * len(tokens_seq)))
        timer.end()
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return seqs, seq_masks, seq_segments, labels

    def trunate_and_pad(self,
                        seq,
                        max_seq_len: int):
        """
        1. 因為本類處理的是單句序列，按照BERT中的序列處理方式，需要在輸入序列頭尾分別拼接特殊字元'CLS'與'SEP'，
           因此不包含兩個特殊字元的序列長度應該小於等於max_seq_len-2，如果序列長度大於該值需要那麼進行截斷。
        2. 對輸入的序列 最終形成['CLS',seq,'SEP']的序列，該序列的長度如果小於max_seq_len，那麼使用0進行填充。

        input:
            seq         : 輸入序列，在本處其為單個句子。
            max_seq_len : 拼接'CLS'與'SEP'這兩個特殊字元後的序列長度

        output:
            seq         : 在入參seq的頭尾分別拼接了'CLS'與'SEP'符號，如果長度仍小於max_seq_len，則使用0在尾部進行了填充。
            seq_mask    : 只包含0、1且長度等於seq的序列，用於表徵seq中的符號是否是有意義的，如果seq序列對應位上為填充符號，
                          那麼取值為1，否則為0。
            seq_segment : shape等於seq，因為是單句，所以取值都為0。

        """
        # 對超長序列進行截斷
        if len(seq) > (max_seq_len - 2):
            seq = seq[0:(max_seq_len - 2)]
        # 分別在首尾拼接特殊符號
        seq = ['[CLS]'] + seq + ['[SEP]']
        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根據max_seq_len與seq的長度產生填充序列
        padding = [0] * (max_seq_len - len(seq))
        # 創建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 創建seq_segment
        seq_segment = [0] * len(seq) + padding
        # 對seq拼接填充序列
        seq += padding
        return seq, seq_mask, seq_segment
