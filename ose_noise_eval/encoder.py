from typing import Dict, Tuple, Optional, Iterator
from collections import Counter
import numpy as np
import json

class Encoder:
    def __init__(self, lower: bool = True, ngrams: int = 2):
        self.vocab: Dict[str, int] = {"[UNK]": 0}
        self.use_ngrams = ngrams
        self.lower = lower
        self._fitted = False

    def toi(self, token: str) -> int:
        """

        :param token:
        :return:

        >>> e = Encoder()
        >>> e.toi("a") == 1
        True
        >>> e._fitted = True
        >>> e.toi("a") == 1 and e.toi("b") == 0
        True
        """
        if token not in self.vocab:
            if self._fitted:
                return 0  # UNK
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def featurize_gt(self, line: Tuple[int, str]) -> Tuple[Optional[int], Dict[int, float]]:
        """

        :param line:
        :return:

        >>> e = Encoder(ngrams=1)
        >>> e.featurize_gt((0, "aaab")) # Uses NGram
        (0, {1: 0.75, 2: 0.25})
        >>> e = Encoder(ngrams=2)
        >>> e.featurize_gt((0, "aaab")) # Uses NGram
        (0, {1: 0.75, 2: 0.25, 3: 0.5, 4: 0.25})
        """
        cls = None
        if isinstance(line, tuple):
            cls, line = line
        matrice = self._line_to_counter(line)
        return cls, matrice

    def _line_to_counter(self, line: str) -> Dict[int, float]:
        if self.lower:
            line = line.lower()
        grams = Counter(line)

        if self.use_ngrams > 1:
            for ngram_size in range(2, self.use_ngrams+1):
                grams += Counter(zip(*[line[i:] for i in range(ngram_size)]))

        # ToDo: BiGrams are divided by the size of the line and not the amount of bigrams in the line
        return {self.toi(tok): cnt / len(line) for tok, cnt in grams.items()}

    def line_to_array(self, line: str) -> np.array:
        mat = self._line_to_counter(line)
        return np.array([mat.get(key, .0) for key in range(self.size())])

    def encode_gt(self, data: Iterator[Tuple[int, str]]) -> Iterator[Tuple[int, Dict[int, float]]]:
        for line in data:
            if len(line[1]):
                yield self.featurize_gt(line)

    def fit(self, data: Iterator[Tuple[int, str]], keep_min: int = 10, limit=300) -> None:
        cnts = Counter()
        for _, line in data:
            if not line.strip():
                continue
            if self.lower:
                line = line.lower()
            line = Counter(line) + Counter(zip(*[line[i:] for i in range(2)]))
            cnts += line

        self._fitted = True
        vocab = [(k, v) for k, v in cnts.items() if v > keep_min and k != "UNK"]
        if limit is None:
            vocab = [k for k, _ in vocab]
        else:
            vocab = [k for k, _ in sorted(vocab, key=lambda x: x[1], reverse=True)][:limit]
        self.vocab = {"[UNK]": 0, **{k: i + 1 for i, k in enumerate(vocab)}}

    def size(self) -> int:
        return len(self.vocab)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(dict(lower=self.lower, ngrams=self.use_ngrams, vocab=self.vocab), f)

    @classmethod
    def load(cls, filepath: str) -> "Encoder":
        with open(filepath) as f:
            j = json.load(f)
        o = cls(lower=j["lower"], ngrams=j["ngrams"])
        o.vocab = j["vocab"]
        o._fitted = True
        return o
