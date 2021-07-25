from typing import List, Tuple, Dict, Optional, Union, Iterator, TextIO
from collections import Counter

import numpy as np
import tqdm

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from .encoder import Encoder

from IPython.display import display, HTML

#enc = Encoder()
#enc.fit(train)
#train_enc = list(enc.encode(train))
#dev_enc = list(enc.encode(dev))
#test_enc = list(enc.encode(test))

GT_Tuple = Tuple[int, str]


class NoiseModel:
    def __init__(self, encoder: Encoder, models: List[LinearClassifierMixin]):
        self.encoder: Encoder = encoder
        self.models: List[LinearClassifierMixin] = models
        if not models:
            # Initiate with default known working models
            self.models = [LogisticRegression(random_state=0), GaussianNB(), RandomForestClassifier()]

    def fit(self, data: List[Tuple[int, Dict[int, float]]]):
        """ Uses the output of encoder.gt_encode()

        :param data:
        :return:
        """
        for model in self.models:
            model.fit(*self.get_arrays(data, encoder=self.encoder))

    def test(self, x, y, raw, name):
        score = []
        bads = {0: [], 1: []}

        for inp, gt, *preds in zip(raw, y, *[mod.predict(x) for mod in self.models]):
            pred, _ = Counter(preds).most_common(1)[0]
            score.append(int(pred == gt))
            if pred != gt:
                bads[gt].append(inp[1])

        score = score.count(1) / len(score)
        return {name: score}, bads

    @staticmethod
    def get_arrays(data: List[Tuple[int, Dict[int, float]]], encoder: Encoder) -> Tuple[List[np.array], np.array]:
        x, y = [], []
        for cls, mat in data:
            x.append(np.array([mat.get(key, .0) for key in range(encoder.size())]))
            y.append(cls)
        return x, np.array(y)

    @staticmethod
    def test_algo(
            model: LinearClassifierMixin,
            x: List[np.array], y: np.array,
            raw: List[GT_Tuple],
            name: Optional[str] = None
    ):
        score = model.score(x, y)
        bads = {0: [], 1: []}

        for inp, pred, gt in zip(raw, model.predict(x), y):
            if pred != gt:
                bads[gt].append(inp[1])

        return {name or str(type(model)): score}, bads

    @staticmethod
    def errors_to_html(errors, name) -> Iterator[str]:
        keys = ['Correct OCR', "Noise"]
        yield f"<h2>{name}</h2>"
        yield "<h3>Bad predictions (category shown is the prediction)</h3>"
        for key, vals in errors.items():
            lis = " ".join([f'<li>{s}</li>' for s in vals])
            yield f"<h4>{keys[key]}</h4><ul>{lis}</ul>"

    def predict_line(self, line: str) -> int:
        x = np.array([self.encoder.line_to_array(line)])
        pred, _ = Counter([mod.predict(x)[0] for mod in self.models]).most_common(1)[0]
        return pred

    def _pred_group(self, sents: List[str], x: np.array):
        for sent, *preds in zip(sents, *[mod.predict(x) for mod in self.models]):
            pred, _ = Counter(preds).most_common(1)[0]
            yield sent, pred

    def predict_file(self,
                     f: TextIO,
                     batch_size: int = 16,
                     verbose: bool = True
    ) -> Tuple[List[Tuple[str, int]], float]:
        output = []

        def to_output(batch):
            x = np.array([encoded for _, encoded in batch])
            for x, y in self._pred_group([s for s, _ in batch], x):
                output.append((x, y))

        batch = []
        if verbose:
            f = tqdm.tqdm(f)

        for line in f:
            if not line.strip():
                continue
            batch.append((line, self.encoder.line_to_array(line)))

            if len(batch) == batch_size:
                to_output(batch)
                batch = []

        if batch:
            to_output(batch)

        return output, sum([1 for _, pred in output if pred == 0]) / max(len(output), 1)


if __name__ == "__main__":
    import glob

    for file in glob.glob("raw/archive.org/**/*.txt"):
        f = open(file)
        sentences, score = predict_file(f)
        print(file, score)
        f.close()
