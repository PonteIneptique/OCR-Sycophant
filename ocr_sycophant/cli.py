import click
import csv

from ocr_sycophant.model import NoiseModel
from ocr_sycophant.encoder import Encoder
from ocr_sycophant.utils import get_dataset


@click.group()
def cli():
    """OCR Simple Noise Evaluator"""


@cli.command("predict")
@click.argument("model", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("files", type=click.File(), nargs=-1)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--logs", default=None, type=click.File(mode="w"))
def predict(model, files, verbose, logs):
    click.secho(click.style(f"Loading model at {model}"))
    model = NoiseModel.load(model)
    click.secho(click.style(f"-> Loaded", fg="green"))
    click.secho(click.style(f"Testing {len(files)} files"))

    def color(score):
        if score >= 0.80:
            return "green"
        else:
            return "red"

    if logs:
        writer = csv.writer(logs)
        writer.writerow(["path", "score"])

    for file in files:
        sentence, clean_score = model.predict_file(file, verbose=verbose)
        click.secho(click.style(f"---> {file.name} has {clean_score*100:.2f}% clean lines", fg=color(clean_score)))
        if logs:
            writer.writerow([file.name, f"{clean_score*100:.2f}"])




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
