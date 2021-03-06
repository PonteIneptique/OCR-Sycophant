import click
import csv
import os
import glob
from multiprocessing import Pool


from ocr_sycophant.model import NoiseModel
from ocr_sycophant.encoder import Encoder
from ocr_sycophant.utils import get_dataset


@click.group()
def cli():
    """OCR Simple Noise Evaluator"""


def _predict(args):
    model, file = args
    try:
        sentence, clean_score = model.predict_filepath(file, batch_size=16, verbose=False)
    except Exception:
        print(Exception)
        print(file)
        return file, .0
    return file, clean_score


@cli.command("predict")
@click.argument("model", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("files", type=click.Path(exists=True, file_okay=True, dir_okay=True), nargs=-1)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--logs", default=None, type=click.File(mode="w"))
@click.option("-w", "--workers", default=1, type=int)
def predict(model, files, verbose, logs, workers):

    def gen(f):
        for file in f:
            if os.path.isdir(file):
                yield from glob.glob(os.path.join(file, "**", "*.txt"), recursive=True)
            else:
                yield file

    click.secho(click.style(f"Loading model at {model}"))
    model = NoiseModel.load(model)
    click.secho(click.style(f"-> Loaded", fg="green"))
    click.secho(click.style(f"Testing {len(list(gen(files)))} files"))

    def color(score):
        if score >= 0.80:
            return "green"
        else:
            return "red"

    def gen_with_models(f):
        for i in gen(f):
            yield model, i

    if logs:
        writer = csv.writer(logs)
        writer.writerow(["path", "score"])

    with Pool(processes=workers) as pool:
        # print same numbers in arbitrary order
        for file, clean_score in pool.imap_unordered(_predict, gen_with_models(files)):
            click.secho(click.style(f"---> {file} has {clean_score*100:.2f}% clean lines", fg=color(clean_score)))
            if logs:
                writer.writerow([file, f"{clean_score*100:.2f}"])





@cli.command("train")
@click.argument("trainfile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("savepath", type=click.Path(file_okay=True, dir_okay=False))
@click.option("--testfile", default=None, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Use specific testfile")
@click.option("--html", default=None, type=click.Path(file_okay=True, dir_okay=False),
              help="Save the errors to HTML")
@click.option("--keep-best", default=False, is_flag=True,
              help="Keep a single model (best performing one)")
def train(trainfile, savepath, testfile, html, keep_best):
    """Train a model with TRAINFILE and save it at SAVEPATH"""
    model = NoiseModel(encoder=Encoder())
    if testfile:
        trainfile = (trainfile, testfile)

    (train, train_enc), (test, test_enc) = get_dataset(trainfile, encoder=model.encoder)
    click.secho(click.style(f"Training {len(model.models)} submodels"))
    model.fit(train_enc)
    click.secho(click.style("--> Done.", fg="green"))

    click.secho(click.style("Testing"))
    scores, errors = model.test(*test_enc, test)
    click.secho(click.style(f"--> Accuracy: {list(scores.values())[0]*100:.2f}", fg="green"))

    if keep_best:
        best, best_model = 0, None
        best_errors = []
        for submodel in model.models:
            out, errs = model._test_algo(submodel, *test_enc, raw=test)
            score = list(out.values())[0]
            if score > best:
                best = score
                best_model = submodel
                best_errors = errs
        click.secho(f"Best model: {type(best_model).__name__} ({100*best:.2f})")
        model.models = [best_model]
        errors = best_errors

    if html:
        with open(html, "w") as f:
            body = """<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>OCR Noise Results</title>
</head>
<body>
{}
</body>
</html>"""
            f.write(body.format("\n".join(model.errors_to_html(errors, "Multi"))))

    click.secho(click.style("Saving"))
    model.save(savepath)
    click.secho(click.style("--> Done.", fg="green"))


if __name__ == "__main__":
    cli()
