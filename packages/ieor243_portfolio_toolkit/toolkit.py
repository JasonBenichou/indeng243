# IND ENG 243
# Ian Howard
# 26 April, 2023

# --------------------------------------------------------------------------------------
# Import dependencies
import importlib
importlib.import_module('.pypfopt', __name__)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from .pypfopt import expected_returns

import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "notebook_connected"  # 'plotly_mimetype+notebook'


# --------------------------------------------------------------------------------------
# ASSET SELECTION

def get_LeastCorrRoots(
        returns1,
        returns2,
        max_correlation: float = 0.5,
) -> pd.Series:
    """
    Get the root symbols of least piecewise correlated stock pairs based on *daily* returns.

    :param returns1:        The adjusted daily returns of mkt cap A as a pandas DataFrame.
    :param returns2:        The adjusted daily returns of mkt cap B as a pandas DataFrame.
    :param max_correlation: An exclusive upper bound on correlation between the pairs.
    :return:                All the root symbols of the least correlated pairs conditioned
                                on the upper bound (input by user) as a pandas Series.
    """
    # Merge the returns into a single set
    merged = returns1.merge(returns2, left_index=True, right_index=True)

    # Compute the correlation matrix
    corr_mtx = merged.corr()

    # Create a mask to isolate the upper triangle
    tri_mask = corr_mtx.mask(np.triu(np.ones_like(corr_mtx, dtype=bool)))

    melted_mtx = tri_mask.reset_index().melt(id_vars="index")  # uses index column as identifier
    melted_mtx.columns = ["Ticker 1", "Ticker 2", "Correlation"]  # the new column names
    melted_mtx = melted_mtx.dropna()

    melted_mtx["Correlation"] = np.abs(melted_mtx["Correlation"])  # diversify: |abs| -> minimise correlation
    low_pairs = melted_mtx.sort_values(by="Correlation", ascending=True)  # smallest values 1st
    low_pairs = low_pairs.reset_index(drop=True)

    # Drop the tickers whose correlation > max_correlation
    least_corr = low_pairs[low_pairs["Correlation"] <= max_correlation]
    least_corr = pd.concat([least_corr["Ticker 1"], least_corr["Ticker 2"]])  # join "Ticker 2" on "Ticker 1"
    least_corr = least_corr.drop_duplicates(keep="first").reset_index(drop=True)  # drop duplicate roots
    return least_corr


def get_LeastCorrPrices(
        set1,
        set2,
        set3,
) -> pd.DataFrame:
    """
    Get the price data only for the least piecewise correlated pairs of the input datasets.

    :param set1: The first pandas Series collection of least correlated roots (string values).
    :param set2: The second pandas Series collection of least correlated roots (string values).
    :param set3: The third pandas Series collection of least correlated roots (string values).
    :return:     The adjusted daily prices of the least piecewise correlated pairs as a pandas DataFrame.
    """
    if not isinstance(set1, pd.Series) and isinstance(set2, pd.Series)\
            and isinstance(set3, pd.Series):
        warnings.warn('Each set should be a pandas Series', RuntimeWarning)
    try:
        # Combine all sets of lowest correlated pairs into a single set
        least_corr_set = pd.concat([set1, set2, set3]).drop_duplicates(keep="first")  # only keep 1st of duplicates
        least_corr_roots = least_corr_set.values.tolist()  # list of root symbol names

        # Retrieve only the price data from the master list corresponding to 'least_corr_roots'
        price_data = prices.loc[:, least_corr_roots]  # prices df must be preloaded to memory
    except NameError:
        price_data = None
    if price_data is None:
        raise NameError("NameError: 'prices' undefined. Try confirming all external `.csv` files have been imported"
                        " and merged into a single pandas DataFrame defining the variable 'prices'")
    else:
        pass
    return price_data


def get_HistReturns(
        prices,
        frequency: int = 252,
        log_returns: bool = False,
        sloc: int = 50,
) -> pd.Series:
    """
    Calculate the Compound Annual Growth Rate (CAGR) mean-historical returns and output
        the top range (sliced at <sloc>) of root symbols.

    :param prices:      Pandas DataFrame of adjusted daily closing prices where each row index=datetime
                            and columns=root.
    :param frequency:   The number of time periods in a trading year.
    :param log_returns: Whether to compute using log returns.
    :param sloc:        A slice indicator for the 'stop' location used to extract the first root symbols
                            from a list comprehension for each mkt cap.
    :return:            The top performing range of root symbols by greatest annualised expected
                            returns as a pandas Series.
    """
    returns = expected_returns.mean_historical_return(  # pd.Series of annualised returns
        prices=prices,
        frequency=frequency,
        log_returns=log_returns,
    )
    # Sort returns so that highest returns are at the top
    returns.sort_values(ascending=False, inplace=True)

    # Filter the first entries of the highest returns up to <sloc> for each mkt cap
    scaps = [symb for symb, val in returns.items() if master_roots.get(symb) == "Small"][:sloc]
    mcaps = [symb for symb, val in returns.items() if master_roots.get(symb) == "Mid"][:sloc]
    lcaps = [symb for symb, val in returns.items() if master_roots.get(symb) == "Large"][:sloc]

    # Stack the filtered returns
    top_returns = pd.concat([returns[scaps], returns[mcaps], returns[lcaps]])
    top_returns.name = "Historical Returns"
    return top_returns


def get_CAPMReturns(
        prices,
        market_prices = None,
        risk_free_rate: int = 0.03,
        log_returns: bool = False,
        sloc: int = 50
) -> pd.Series:
    """
    Compute a return estimate using the Capital Asset Pricing Model (CAPM) and output
        the top range (sliced at <sloc>) of root symbols.

    :param prices:         Pandas DataFrame of adjusted daily closing prices where each row index=datetime
                               and columns=root.
    :param market_prices:  Pandas DataFrame of adjusted daily closing prices of the benchmark.
    :param risk_free_rate: The risk-free rate of borrowing/lending. You should use the appropriate time
                               period, corresponding to the frequency parameter.
    :param log_returns:    Whether to compute using log returns.
    :param sloc:           A slice indicator for the 'stop' location used to extract the first root symbols
                               from a list comprehension for each mkt cap.
    :return:               The top performing range of root symbols by greatest annualised expected
                               returns as a pandas Series.
    """
    returns = expected_returns.capm_return(  # pd.Series of annualised returns
        prices,
        market_prices=market_prices,
        risk_free_rate=risk_free_rate,
        log_returns=log_returns,
    )
    # Sort returns so that highest returns are at the top
    returns.sort_values(ascending=False, inplace=True)

    # Filter the first entries of the highest returns up to <sloc> for each mkt cap
    scaps = [symb for symb, val in returns.items() if master_roots.get(symb) == "Small"][:sloc]
    mcaps = [symb for symb, val in returns.items() if master_roots.get(symb) == "Mid"][:sloc]
    lcaps = [symb for symb, val in returns.items() if master_roots.get(symb) == "Large"][:sloc]

    # Stack the filtered returns
    top_returns = pd.concat([returns[scaps], returns[mcaps], returns[lcaps]])
    top_returns.name = "CAPM Returns"
    return top_returns


def get_Volatility(
        daily_returns,
        frequency: int = 252,
        max_vol: float = 1,
        sloc: int = 50,
) -> pd.Series:
    """
    Computes the volatility of each stock from the adjusted daily returns.

    :param daily_returns: The adjusted daily returns as a pandas DataFrame.
    :param frequency:     The number of time periods in a trading year.
    :param max_vol:       An inclusive upper bound on volatility to be included in the output.
    :param sloc:          A slice indicator for the 'stop' location used to extract the first root symbols
                              from a list comprehension for each mkt cap.
    :return:              The top performing range of root symbols by least annualised volatility as a
                              pandas Series.
    """
    # Compute the annualised volatility for each entry
    volatility = daily_returns.std() * (frequency**0.5)

    # Sort the values from lowest to highest
    volatility.sort_values(ascending=True, inplace=True)

    # Drop volatilities >= 'max_vol' value
    volatility_dict = {symb: vol for symb, vol in volatility.items() if vol <= max_vol}

    # Filter the lowest volatilities by some <sloc> amount on mkt cap & retrieve the root symbol
    scaps = [symb for symb, val in volatility_dict.items() if master_roots.get(symb) == "Small"][:sloc]
    mcaps = [symb for symb, val in volatility_dict.items() if master_roots.get(symb) == "Mid"][:sloc]
    lcaps = [symb for symb, val in volatility_dict.items() if master_roots.get(symb) == "Large"][:sloc]

    # Stack the filtered volatilities
    least_vol = pd.concat([volatility[scaps], volatility[mcaps], volatility[lcaps]])
    least_vol.name = "Volatility"
    return least_vol


# --------------------------------------------------------------------------------------
# OPTIMISATION

def risk_model_prices(
        returns,
) -> pd.DataFrame:
    """
    Creates a dataset of adjusted daily prices only for the root symbols included in the
        set of daily returns.

    :param returns: The top performing range of root symbols by greatest annualised expected
                        returns given as a pandas Series.
    :return:        The adjusted daily price data for all roots included in the set of
                        returns as a pandas DataFrame.
    """
    # Filter the first entries of the highest returns up to <sloc> for each mkt cap
    scaps = [symb for symb, val in returns.items() if master_roots.get(symb) == "Small"]
    mcaps = [symb for symb, val in returns.items() if master_roots.get(symb) == "Mid"]
    lcaps = [symb for symb, val in returns.items() if master_roots.get(symb) == "Large"]

    # Merge all the collections into one set
    model_prices = pd.concat([prices[scaps], prices[mcaps], prices[lcaps]], axis=1)
    return model_prices


# --------------------------------------------------------------------------------------
# VISUALISATIONS

def get_SVGHeatMap(
        matrix,
        titleL: str,
        titleR: str,
) -> go.Heatmap:
    """
    Create a static heatmap image using plotly 'graph_objects' 'Figure'.

    :param matrix:  Correlation matrix as pandas DataFrame.
    :param titleL:  The first mkt-cap to be featured in the heatmap title.
    :param titleR:  The second mkt-cap to be featured in the heatmap title.
    :return:        A static SVG image as a plotly correlation heatmap.
    """
    # Create the heatmap using plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.values,
            y=matrix.index.values,
            colorscale="Viridis",  # colour can be modified here according to user preference
        ),
        layout=go.Layout(
            title="<span style='font-size: 18pt;'>Correlation Matrix <b>Heatmap</b></span>"
                  f"<br><span style='font-size: 12pt;'>{titleL} | {titleR}</span>",
            xaxis=dict(title=""),
            yaxis=dict(title=""),
        )
    )
    # Update plotly layout w/ additional customisation
    fig.update_xaxes(tickfont_size=8)
    fig.update_yaxes(tickfont_size=8)
    fig.update_layout(
        title={
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        coloraxis_colorbar=dict(  # defines the colourbar tick values
            tickvals=[np.min(matrix.values), np.max(matrix.values)],
            tickprefix="-",
        )
    )
    # Update colourbar tick font
    fig.update_traces(colorbar_tickfont_size=8, selector=dict(type="heatmap"))

    # Display the plot as svg for GitHub
    heatmap = fig.show("svg")
    return heatmap


def get_LiveHeatMap(
        matrix,
        titleL: str,
        titleR: str,
) -> go.Heatmap:
    """
    Create an interactive heatmap figure using plotly "graph_objects" "Figure". If the figure
        fails to load, the `.html` file can be directly opened in a web browser.

    :param matrix:  Correlation matrix as pandas DataFrame.
    :param titleL:  The first mkt-cap to be featured in the heatmap title.
    :param titleR:  The second mkt-cap to be featured in the heatmap title.
    :return:        An interactive HTML figure as a plotly correlation heatmap.
    """
    # Create the heatmap using plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.values,
            y=matrix.index.values,
            colorscale='Viridis',  # colour can be modified here according to user preference
        ),
        layout=go.Layout(
            title="<span style='font-size: 18pt;'>Correlation Matrix <b>Heatmap</b></span>"
                  f"<br><span style='font-size: 12pt;'>{titleL} | {titleR}</span>",
            xaxis=dict(title=""),
            yaxis=dict(title=""),
        )
    )
    # Update plotly layout w/ additional customisation
    fig.update_xaxes(tickfont_size=8)
    fig.update_yaxes(tickfont_size=8)
    fig.update_layout(
        title={
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            'yanchor': "top",
        },
        coloraxis_colorbar=dict(  # defines the colourbar tick values
            tickvals=[np.min(matrix.values), np.max(matrix.values)],
            tickprefix="-",
        )
    )
    # Update colourbar tick font
    fig.update_traces(colorbar_tickfont_size=8, selector=dict(type="heatmap"))

    # Save the figure locally to the cwd
    fig.write_html(f"{titleL}_{titleR}.html")

    # Import & display the HTML figure
    from IPython.display import HTML
    heatmap = HTML(filename=f"{titleL}_{titleR}.html")
    return heatmap


def plot_TriHeatmap(
        returns1,
        returns2,
        titleL:str,
        titleR:str,
) -> sns.heatmap:
    """
    Computes a correlation matrix and generates the lower triangle as a seaborn heatmap.

    :param returns1: The daily returns of mkt cap A as a pandas DataFrame.
    :param returns2: The daily returns of mkt cap B as a pandas DataFrame.
    :param titleL:   The first mkt-cap to be featured in the heatmap title.
    :param titleR:   The second mkt-cap to be featured in the heatmap title.
    :return:         Lower triangle of correlation matrix as a seaborn heatmap.
    """
    # Merge the returns into a single set
    merged = returns1.merge(returns2, left_index=True, right_index=True)

    # Compute the correlation matrix
    corr_mtx = merged.corr()
    corr_mtx.abs()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_mtx, dtype=bool))

    # Set up the matplotlib figure size
    plt.subplots(figsize=(20, 15))

    # Set a colormap; can be modified here according to user preference
    sns.color_palette("rocket", as_cmap=True)

    # Draw the heatmap with the mask & correct aspect ratio
    hm = sns.heatmap(corr_mtx, mask=mask, cmap="rocket")
    hm.set_xticklabels(hm.get_xticklabels(), fontsize=6)
    hm.set_yticklabels(hm.get_yticklabels(), fontsize=6)
    plt.gca().set_title(
        r"Triangle Correlation $\bf{Heatmap}$",
        fontsize=26,
        loc="center",
        y=1.12,
    )
    hm.text(  # places 2nd row title
        x=.5,
        y=1.115,
        s=f"{titleL} | {titleR}",
        ha="center",
        va="top",
        transform=hm.transAxes,
        fontsize=20,
    )



