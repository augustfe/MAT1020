import seaborn as sns
import numpy as np
import matplotlib as mpl
from pathlib import Path
from pandas import DataFrame
import matplotlib.pyplot as plt


# Set up for LaTeX rendering
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["figure.titlesize"] = 15


class TimeSeriesPlotter:
    month_by_season = np.array(
        [
            None,
            "Winter",
            "Winter",
            "Spring",
            "Spring",
            "Spring",
            "Summer",
            "Summer",
            "Summer",
            "Autumn",
            "Autumn",
            "Autumn",
            "Winter",
        ]
    )

    def __init__(
        self,
        X: DataFrame,
        period: str,
        cmap: str = "coolwarm",
        datalabel: str = None,
        save_dir: Path = None,
    ) -> None:
        """Plotting utilities for time series data

        Args:
            X (DataFrame): Data to plot
            period (str): Period of the data
            cmap (str, optional): Color map for heatmaps. Defaults to "coolwarm".
            datalabel (str, optional): Label for data. Defaults to None.
        """
        self.X = X
        self.period = period
        self.cmap = cmap
        self.datalabel = datalabel
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        self.save = save_dir

    def plot_heatmap(
        self,
        X: DataFrame,
        title: str,
        mask: np.ndarray = None,
    ) -> None:
        """Plot a heatmap of the data

        Args:
            X (DataFrame): Data to plot
            title (str): Title of the plot
            mask (np.ndarray, optional): Whether to mask the plot or not. Defaults to None.
        """
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            X,
            cmap=self.cmap,
            annot=True,
            square=True,
            linewidth=0.5,
            mask=mask,
        )
        ax.set_title(title)
        if self.save:
            plt.savefig(self.save / f"{title}.pdf", bbox_inches="tight")

        ax.plot()

    def plot_df(self, X: DataFrame, title: str) -> None:
        """Plot a DataFrame

        Args:
            X (DataFrame): Data to plot
            title (str): Title of the plot
        """
        plt.figure(figsize=(8, 6))
        ax = X.plot(title=title, ylabel=self.datalabel)
        if self.save:
            plt.savefig(self.save / f"{title}.pdf", bbox_inches="tight")
        ax.plot()

    def plot_raw(self) -> None:
        """Plot the raw data"""
        title = f"{self.datalabel} {self.period}"
        self.plot_df(self.X, title)

    def cumsum_rolling(self, year: int = None, window: int = 30) -> None:
        """Plot the rolling mean and cumulative sum of the data

        Args:
            year (int, optional): Year of the data. Default to None.
            window (int, optional): Window size for rolling mean. Defaults to 30.
        """
        X, base_title = self.choose_data(year)
        rolling_title = f"{base_title} - Rolling {window} day"
        self.plot_df(X.rolling(window=window).mean(), rolling_title)

        cumsum_title = f"{base_title} - Cumulative Sum"
        self.plot_df(X.cumsum(), cumsum_title)

    def mean_by_season(self, year: int = None) -> None:
        """Plot the mean of the data by season

        Args:
            year (int, optional): Year of the data. Default to None.
        """
        X, base_title = self.choose_data(year)
        season_title = f"{base_title} - Mean by Season"

        groups = self.month_by_season[X.index.month]
        X_season = X.copy()
        X_season.insert(0, "Season", groups)

        self.plot_heatmap(X_season.groupby(["Season"]).mean(), season_title)

    def cov(self, year: int = None) -> None:
        """Plot the covariance of the data

        Args:
            year (int, optional): Year of the data. Default to None.
        """
        X, base_title = self.choose_data(year)
        cov = X.cov()
        cov_title = f"{base_title} - Covariance"

        mask = np.triu(np.ones_like(cov, dtype=bool), k=1)
        self.plot_heatmap(cov, cov_title, mask=mask)

    def corr(
        self,
        year: int = None,
    ) -> None:
        """Plot the correlation of the data

        Args:
            year (int, optional): Year of the data. Default to None.
        """
        X, base_title = self.choose_data(year)
        corr = X.corr()
        corr_title = f"{base_title} - Correlation"

        mask = np.triu(np.ones_like(corr, dtype=bool))

        self.plot_heatmap(corr, corr_title, mask=mask)

    def choose_data(self, year: int | None) -> None:
        """Pick data to plot.

        Args:
            year (int, None): Year to plot
        """
        if year:
            data = self.X[self.X.index.year == year]
            base_title = f"{self.datalabel} {year}"
        else:
            data = self.X
            base_title = f"{self.datalabel} {self.period}"
        return data, base_title
