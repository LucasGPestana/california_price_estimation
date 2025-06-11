import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import pandas as pd


from sklearn.metrics import PredictionErrorDisplay


from typing import Iterable, Union, Sequence

def compositionHistBox(data: pd.DataFrame, columns: Iterable[str], bins: Union[str, int]="sturges"):

    for column in columns:

        fig, axs = plt.subplots(
            nrows=2, 
            ncols=1, 
            sharex=True, 
            figsize=(8, 6)
            )

        sns.histplot(
            data=data,
            x=column,
            kde=True,
            bins=bins,
            ax=axs[1],
        )

        sns.boxplot(
            data=data,
            x=column,
            ax=axs[0],
        )

        fig.suptitle(column)
    
    plt.tight_layout()
    plt.show()

def compareMetrics(df: pd.DataFrame, metrics_names: Sequence[str]) -> None:

    metrics = df.columns.drop(["model"])

    if len(metrics) != len(metrics_names):

        raise Exception("A quantidade de métricas identificadas no dataframe e os nomes das métricas não batem!")

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex="col", figsize=(10, 10))

    for ax, metric, metric_name in zip(axs.flatten(), metrics, metrics_names):

        sns.boxplot(
            data=df,
            y=metric,
            x="model",
            ax=ax
        )

        ax.set_ylabel(metric_name)

        ax.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.show()

def plotCoeffs(df_coeffs: pd.DataFrame) -> None:

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.barplot(
        x=df_coeffs["coeficientes"].values,
        y=df_coeffs.index,
        orient='h',
        ax=ax
    )

    ax.axvline(x=0, linewidth=1, color="black")

    ax.set_ylabel("Features")
    ax.set_xlabel("Coeficientes")

    plt.show()

def plotResiduals(y_true, y_pred, eng_formatter: bool=False, sample_size: float=0.25) -> None:

    residuals = y_true - y_pred

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    sns.histplot(
        x=residuals,
        kde=True,
        ax=axs[0]
    )

    PredictionErrorDisplay.from_predictions(
        y_true, 
        y_pred,
        kind="residual_vs_predicted",
        subsample=sample_size,
        random_state=42,
        ax=axs[1],
        scatter_kwargs={"alpha": 0.2}
        )
    
    PredictionErrorDisplay.from_predictions(
        y_true, 
        y_pred,
        kind="actual_vs_predicted",
        subsample=sample_size,
        random_state=42,
        ax=axs[2],
        scatter_kwargs={"alpha": 0.2}
        )
    
    if eng_formatter:

        for ax in axs:

            ax.xaxis.set_major_formatter(EngFormatter())
            ax.yaxis.set_major_formatter(EngFormatter())

    plt.tight_layout()
    plt.show()

def plotResidualsFromEstimator(estimator, X, y, eng_formatter: bool=False, sample_size: float=0.25) -> None:

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 8))

    pred_error_residuals = PredictionErrorDisplay.from_estimator(
        estimator, 
        X,
        y,
        kind="residual_vs_predicted",
        subsample=sample_size,
        random_state=42,
        ax=axs[1],
        scatter_kwargs={"alpha": 0.2}
        )
    
    _ = PredictionErrorDisplay.from_estimator(
        estimator, 
        X,
        y,
        kind="actual_vs_predicted",
        subsample=sample_size,
        random_state=42,
        ax=axs[2],
        scatter_kwargs={"alpha": 0.2}
        )
    
    residuals = pred_error_residuals.y_true - pred_error_residuals.y_pred

    sns.histplot(
        x=residuals,
        kde=True,
        ax=axs[0]
    )

    if eng_formatter:

        for ax in axs:

            ax.xaxis.set_major_formatter(EngFormatter())
            ax.yaxis.set_major_formatter(EngFormatter())

    plt.tight_layout()
    plt.show()

    

