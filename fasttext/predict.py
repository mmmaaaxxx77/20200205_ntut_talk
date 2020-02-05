"""Demo for FastText model."""
import traceback

from gensim import models

import jieba


class Demo():

    def __init__(self):
        self.ft_model = models.FastText.load("/Users/johnny/Desktop/version_1.6.3/FastText_v1.6.3/ft.model.bin")
        self.w2v_model = models.Word2Vec.load("/Users/johnny/Desktop/version_1.6.3/Word2Vec_v1.6.3/w2v.model.bin")

    def demo(self):
        print('Start Demo')
        while True:
            print('Input a word, get top 20 similar words.')
            try:
                query = input()
            except:
                print("Input Error.")
            if query == 'quit':
                raise ValueError

            try:
                q_list = query.split()
                if len(q_list) == 1:
                    q = q_list[0]
                    print('Top 20 similar word: ')

                    print("#### FastText ####")

                    # FastText
                    res = self.ft_model.most_similar(q, topn=10)
                    # print(f'{self.ft_model[q]}')
                    for idx, item in enumerate(res):
                        jws = [j for j in jieba.cut(item[0])]
                        print(f"{item[0]} / {item[1]} / {jws}")

                    print("#### W2V ####")

                    # W2V
                    res = self.w2v_model.most_similar(q, topn=10)
                    # print(f'{self.ft_model[q]}')
                    for idx, item in enumerate(res):
                        jws = [j for j in jieba.cut(item[0])]
                        print(f"{item[0]} / {item[1]} / {jws}")
            except:
                print(traceback.format_exc())
                print('End Demo')


if __name__ == '__main__':
    d = Demo()
    d.demo()
