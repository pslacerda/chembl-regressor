from molfeat.calc import FP_FUNCS, FPCalculator
import click
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from lightgbm import LGBMRegressor
import seaborn as sb
from matplotlib import pyplot as plt



def parse_fpt(smis):
    res = []
    for smi in smis:
        row_res = {}
        for method in FP_FUNCS:
            if method == "map4":
                continue
            fpc = FPCalculator(method)
            fp = fpc(smi)
            for idx, bit in enumerate(fp):
                row_res[f"{method}_{idx}"] = bit
        res.append(row_res)
    return pd.DataFrame(res)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--test-size", default=0.2, help="fraction of test size")
@click.option("--random-seed", default=0)
@click.option("--cross-val", default=10, help="number of cross validation folds")
@click.option("--drop-na/--no-drop-na", default=False, help="drop NaN values")
@click.option("--n-jobs", default=-1, help="number of jobs")
@click.option(
    "--target-var",
    default="pChEMBL Value",
    type=click.Choice(["pChEMBL Value", "Standard Value"]),
    help="target variable",
)
@click.option(
    "--sample-fraction",
    default=1.0,
    help="use only a sample fraction of the input data",
)
@click.option(
    "--log10/--no-log10",
    default=False,
    help="may be useful for standartize the target variable",
)
@click.option("--plot/--no-plot", default=True, help="display plots")
@click.option(
    "--output-model",
    default="regressor.joblib",
    type=click.Path(exists=False),
    help="output model filename",
)
@click.argument("chembl-input", type=click.Path(exists=True))
def train(
    test_size,
    random_seed,
    cross_val,
    drop_na,
    n_jobs,
    target_var,
    sample_fraction,
    log10,
    plot,
    output_model,
    chembl_input,
):
    """This program train a regressor based on ChEMBL dataset."""
    data = pd.read_csv(
        chembl_input,
        na_values=["", "None"],
        sep=";",
        usecols=["Smiles", "Smiles", target_var],
    )
    data = data.sample(frac=sample_fraction, random_state=random_seed)

    y = data[target_var]
    y = y.rename("y")

    if log10:
        y = np.log10(y)

    if np.any(pd.isna(data)) and not drop_na:
        num = np.sum(pd.isna(data), axis=0)
        click.secho(f"There are NaN values into your dataset.", fg="red", err=True)
        click.secho(num, fg="red", err=True)
        return

    if drop_na:
        click.echo("Cleaning NaN values... ", nl=False)
        idx = ~np.any(pd.isna(data), axis=1)
        data = data[idx]
        y = y[idx]

        idx = ~pd.isna(y)
        data = data[idx]
        y = y[idx]

        ratio = 100 * np.sum(idx) / len(idx)
        click.secho(f"About {ratio:.2f}% of data remained.", fg="green")

    click.secho(f"Max {target_var}: {np.max(y)}, min {target_var}: {np.min(y)}")
    if np.max(y) - np.min(y) < 3:
        click.secho(f"Short range of {target_var}.", fg="yellow")

    click.echo("Computing fingerprints... ", nl=False)
    smis = data.pop("Smiles")
    fpts = parse_fpt(smis)
    click.secho("Done.", fg="green")

    X_train, X_test, y_train, y_test = train_test_split(
        fpts, y, test_size=test_size, random_state=random_seed
    )

    click.secho(f"Train size: {len(X_train)}")
    click.secho(f"Test size: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    for model in [
        LGBMRegressor(random_state=random_seed, verbose=-1),
        RandomForestRegressor(random_state=random_seed),
    ]:
        label = model.__class__.__name__
        click.echo(f"Fitting model {label}... ", nl=False)
        cv = GridSearchCV(model, {}, cv=cross_val, n_jobs=n_jobs)
        cv.fit(X_train_scaled, y_train)
        click.secho("Done.", fg="green")

        best_model = cv.best_estimator_

        y_pred = best_model.predict(X_test_scaled)
        corr = pearsonr(y_test, y_pred)
        click.secho(f"{label}: ", bold=True, nl=False)
        click.secho(f"corr={corr[0]:.2f}, p={corr[1]:.2f}", fg="green", bold=True)

        models[label] = (best_model, corr[0])

    best_model = sorted(models.values(), key=lambda m: m[1], reverse=True)[0][0]
    joblib.dump((scaler, best_model), output_model)

    if plot:
        y_pred = best_model.predict(X_test_scaled)  # again

        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)

        sb.residplot(x=y_test, y=y_pred, color="g", ax=ax1)
        ax1.set(label="Residuals", ylabel="Residuals", xlabel="Predicted Value")

        sb.regplot(x=y_test, y=y_pred, ax=ax2)
        ax2.set(label="Regression", xlabel="y test", ylabel="y pred")

        plt.tight_layout()
        plt.show()


@cli.command()
@click.option("--input-model", default="regressor.joblib", help="output model filename")
@click.option("--output-scores", default="scores.tsv", type=click.Path())
@click.argument("compounds_smi", type=click.Path(exists=True))
def predict(input_model, compounds_smi, output_scores):
    """This program predict the pChEMBL value for a list of compounds."""
    click.echo("Loading model.")
    (scaler, model) = joblib.load(input_model)

    click.echo("Reading compounds.")
    data = pd.read_csv(compounds_smi, sep="\t", names=("Smiles", "Name"))

    click.echo("Computing results...")
    
    with open(output_scores, "w") as output_file:
        for idx, row in data.iterrows():
            try:
                fpt = parse_fpt([row["Smiles"]])
                X = scaler.transform(fpt)
                y = model.predict(X)[0]
                print(row["Name"], round(y, 2), sep="\t", file=output_file)
                output_file.flush()
            except Exception as e:
                click.echo(f"Error on {row['Name']}: {e}", err=True)


if __name__ == "__main__":
    cli()
