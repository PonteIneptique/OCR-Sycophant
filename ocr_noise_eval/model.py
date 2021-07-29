from typing import List, Tuple, Dict, Optional, Union, Iterator, TextIO
from collections import Counter

import joblib
import numpy as np
import tqdm

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from ocr_noise_eval.encoder import Encoder


GT_Tuple = Tuple[int, str]


class NoiseModel:
    def __init__(self, encoder: Encoder, models: List[LinearClassifierMixin] = []):
        self.encoder: Encoder = encoder
        self.models: List[LinearClassifierMixin] = models
        if not models:
            # Initiate with default known working models
            self.models = [
                LogisticRegression(random_state=0), GaussianNB(), RandomForestClassifier()]

    def fit(self, data: List[Tuple[int, Dict[int, float]]]):
        """ Uses the output of encoder.gt_encode()

        :param data:
        :return:
        """
        for model in tqdm.tqdm(self.models):
            model.fit(*data)

    def test(self, x, y, raw):
        score = []
        bads = {0: [], 1: []}

        for inp, gt, *preds in zip(raw, y, *[mod.predict(x) for mod in self.models]):
            pred, _ = Counter(preds).most_common(1)[0]
            score.append(int(pred == gt))
            if pred != gt:
                bads[gt].append(inp[1])

        score = score.count(1) / len(score)
        return {type(self).__name__: score}, bads

    @staticmethod
    def _test_algo(
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
        """

        :param f:
        :param batch_size:
        :param verbose:
        :return: Lines that needs to be kept, CLeanness score (The higher the best)
        """
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

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


if __name__ == "__main__":
    model = NoiseModel(encoder=Encoder())
    from ocr_noise_eval.utils import get_dataset
    (train, train_enc), (test, test_enc) = get_dataset("dataset.csv", encoder=model.encoder)

    model.fit(train_enc)
    print("Fitted")
    scores, errors = model.test(*test_enc, test)
    print(scores)
    print("\n".join(model.errors_to_html(errors, "Multi")))

    model.save("model.saved")
    model = NoiseModel.load("model.saved")
    scores, errors = model.test(*test_enc, test)
    print(scores)

    import glob

    for file in glob.glob("../new-latin-bert/raw/archive.org/**/*.txt"):
        f = open(file)
        sentences, score = model.predict_file(f)
        print(file, score)
        f.close()
