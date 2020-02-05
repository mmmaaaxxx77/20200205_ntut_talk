from gensim.models import FastText
from gensim.models import word2vec


class FastTextTrainner:
    """FastText model's trinner."""

    def __init__(self):
        self.train_data_path = "content/text.txt"
        self.model_path = "model/ft.model.bin"

    def train(self):
        """Gen FastText model."""

        print(f'Build sentences from {self.train_data_path}')
        sentences = word2vec.Text8Corpus(self.train_data_path)

        model = FastText(
            sentences,
            size=100,
            window=5,
            min_count=10,
            workers=6,
        )
        model.save(self.model_path)

        print(
            f'Finish train model FastText\n'
            f'save to {self.model_path}'
        )


if __name__ == '__main__':
    w = FastTextTrainner()
    w.train()
