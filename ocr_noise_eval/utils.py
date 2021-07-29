from typing import Union, Tuple, TYPE_CHECKING
import csv
import random

if TYPE_CHECKING:
    import ocr_noise_eval.encoder as enc


def _read_file(path):
    datas = []
    with open(path) as f:
        r = csv.DictReader(f)
        for l in r:
            line = (int(l.get("noise", "").strip() in {"1", "2"}), l["sentence"])
            if line not in datas:
                datas.append(line)
    random.shuffle(datas)
    return datas


def get_dataset(
        paths: Union[str, Tuple[str, str]],
        encoder: "enc.Encoder",
        ratio: float = 0.8,
        fit=True,
        fit_kwargs=None
):
    if isinstance(paths, tuple):
        train, test = _read_file(paths[0]), _read_file(paths[1])
    else:
        datas = _read_file(paths)
        train = int(len(datas) * ratio)
        train, test = datas[:train], datas[train:]

    if fit:
        fit_kwargs = fit_kwargs or dict(keep_min=10, limit=300)
        encoder.fit(train, **fit_kwargs)

    train_enc, test_enc = encoder.encode_gt(train), encoder.encode_gt(test)

    return (train, train_enc), (test, test_enc)
