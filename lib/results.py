"""
Author: Guillermo Romero Moreno (Guillermo.RomeroMoreno@ed.ac.uk)
Date: 9/2/2022

This file contains the functions to reproduce results from the paper.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges, \
    LabelSet
from bokeh.palettes import Spectral8
from bokeh.plotting import figure, from_networkx

from lib.data import beautify_index, stratify_dataset
from lib.model import MLTCModel


def fit_model(data: pd.DataFrame, strat_var: str, index, model_class: MLTCModel, num_warmup: int = 500,
              model_name: str = None, sample_kwargs: dict = {}, **model_kwargs) -> MLTCModel:
    """
    Fit a Bayesian model to a specific group

    :param data: in long-form format (i.e. patients as rows and variables as columns)
    :param strat_var: variable to use for stratification. See the `data.stratify_dataset` function for options.
    :param index: (int or tuple) index of the strata to use
    :param model_class: class to use as a model
    :param num_warmup: number of warmup iterations to use in model fitting
    :param model_name: name of the model (str), corresponding to the '.stan' file to be loaded
    :param sample_kwargs: additional keyword arguments to pass to the *Stan* sampling function
    :param **model_kwargs: additional keyword arguments to pass to the model class' `__init__` function
    :return: fitted MLTC model
    """
    dfs, labels, fnames = stratify_dataset(data, strat_var)

    if model_name is not None:
        model_kwargs["model_name"] = model_name
    model = model_class(**model_kwargs)

    assert strat_var != "Age-Sex" or (type(index) == tuple and len(index) == 2)
    data = dfs[index] if strat_var != "Age-Sex" else dfs[2 * index[0] + index[1]]
    fname = fnames[index] if strat_var != "Age-Sex" else fnames[index[0]][index[1]]

    model.load_fit(data, fname + "0mu" + model.create_file_name(num_warmup), column_names=names,
                   num_warmup=num_warmup, random_seed=1, **sample_kwargs)
    model.plot_training_statistics(separate_chains=False)
    model.plot_logprob(separate_chains=True, temporal=True)

    return model


def get_top_associations(assoc_df: pd.DataFrame, columns: list = ["a_f_sig_CI", "RR_f_sig_CI"],
                         top_N_positive: int = 10, top_N_negative: int = None, **beautify_index_kwargs):
    """
    From an association dataframe, return the top 10 significant associations for ABC and RR and the negative
    associations for each of these measures.

    :param assoc_df: pandas Dataframe with association information.
    :param columns: (list of str) columns to keep in the returned tables
    :param top_N_positive: number of rows to include in the top 10 significant associations
    :param top_N_negative: number of rows to include in the top 10 significant negative associations
    :return: four pandas dataframes: top 10 ABCs, negative ABCs, top 10 RRs, negative RRs
    """
    top_N_positive = top_N_positive if top_N_positive is not None else len(assoc_df)
    top_N_negative = top_N_negative if top_N_negative is not None else len(assoc_df)
    top_a = assoc_df[assoc_df["a_sig"].astype(bool)].sort_values("ABC", ascending=False).iloc[:top_N_positive][columns]
    neg_a = assoc_df[assoc_df["a_sig"].astype(bool) & (assoc_df["ABC"] < 1)].sort_values("ABC", ascending=True).iloc[
            :top_N_negative][columns]
    top_RR = assoc_df[assoc_df["fisher_sig"].astype(bool)].sort_values("RR", ascending=False).iloc[:top_N_positive][
        columns[::-1]]
    neg_RR = assoc_df[assoc_df["fisher_sig"].astype(bool) & (assoc_df["RR"] < 1)]
    neg_RR = neg_RR.sort_values("RR", ascending=True).iloc[:top_N_negative][columns[::-1]]

    return [beautify_index(tab, **beautify_index_kwargs) for tab in (top_a, neg_a, top_RR, neg_RR)]


def plot_tops(data: pd.DataFrame, ax: plt.Axes, panel_label: str = None, NS_pos=1, y_dict: dict = {"RR": {}, "ABC": {}},
              negative: bool = False, top_N: int = 10):
    """
    Produce plots ot the top associations for ABC and RR and how they are related.

    :param data: pandas Dataframe with association information.
    :param ax: plt.Axes object onto which to plot
    :param panel_label: label to add to the panel
    :param NS_pos: (float) y-position of the "ABC not significant" label
    :param y_dict: dictionary with keys "RR" and "ABC" and values sub-dictionaries containing specific association pairs
    and the y-displacement of their labels
    :param negative: whether the top 10 negative associations should be plotted
    :param top_N: number of associations to include
    """
    # gather associations
    tops = get_top_associations(data, columns=["RR", "ABC", "a_sig", "RR_f_sig_CI", "a_f_sig_CI", "RR_conf_down",
                                               "RR_conf_up", "a_conf_down", "a_conf_up"], top_N_negative=20, short=True)
    if not negative:
        assocs = pd.concat([tops[2], tops[0]])  # Top 10 RR and top 10 ABC
    else:
        tops = [tops[3], tops[1]]  # Bottom 20 RR and bottom 20 ABC
        for res in tops:
            res["a_f"] = res["ABC"].apply(lambda x: f"{x:.2f}")
            res["a CI (99%)"] = res["a_conf_down"].apply(lambda x: f"{x:.2f}") + " - " + res["a_conf_up"].apply(
                lambda x: f"{x:.2f}")
            res["a_f_sig_CI"] = res["a_f"] + " (" + res["a CI (99%)"] + ")"

            res["RR_f"] = res["RR"].apply(lambda x: f"{x:.2f}")
            res["RR CI (99%)"] = res["RR_conf_down"].apply(lambda x: f"{x:.2f}") + " - " + res["RR_conf_up"].apply(
                lambda x: f"{x:.2f}")
            res["RR_f_sig_CI"] = res["RR_f"] + " (" + res["RR CI (99%)"] + ")"

        assocs = pd.concat(tops)
    assocs = assocs[~assocs.index.duplicated(keep='first')]

    anno_kwargs = dict(va="center", textcoords="offset points")
    for i, (label, row) in enumerate(assocs.iterrows()):
        # plot lines
        ax.plot([0, 1], [row["RR"], row["ABC"] if row["a_sig"] else NS_pos], marker="o")

        # print labels. The two lines below assume that all associations found by ABC are also found by RR. Otherwise,
        # you will need to adjust the code
        ax.annotate(f'{label}, RR {row["RR_f_sig_CI"]}', (0, row["RR"]),
                    xytext=(-10, y_dict["RR"][label] if label in y_dict["RR"] else 0), ha='right',
                    color="black" if i < top_N else "grey", **anno_kwargs)
        ax.annotate(f'ABC {row["a_f_sig_CI"]}, {label}', (1, row["ABC"]),
                    xytext=(10, y_dict["ABC"][label] if label in y_dict["ABC"] else 0), ha='left',
                    **anno_kwargs) if row["a_sig"] else None
        ax.annotate("ABC not significant", (1, NS_pos), xytext=(10, 0), ha='left', **anno_kwargs)

        ax.set(yscale="log", ylim=[None, 1] if negative else 0.9 * assocs[assocs["a_sig"] == True]["ABC"].min())
        ax.spines['top'].set_visible(negative)
        ax.spines['bottom'].set_visible(not negative)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([0, 1], ["Top ten RR", "Top ten ABC"] if not negative else ["Negative RR", "Negative ABC"])
        ax.set_yticks([])
        if panel_label is not None:
            ax.annotate(panel_label, xy=(-3.05, 1.), xycoords='axes fraction', fontweight='bold', fontsize=15)

        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False) if negative else None
        ax.minorticks_off()


def plot_ABC_vs_RR(ax: plt.Axes, dat: pd.DataFrame, ABC_type: str = "ABC", RR_conf: bool = True, a_conf: bool = True,
                   colorbar: bool = False, log: bool = True, xlim=None, ylim=None, title: str = None):
    """
    Scatter plot showing the relationship between associations with RR (x-axis) and ABC (y-axis), coloured by the
    prevalence of the condition with the highest prevalence of the pair.

    :param ax: axis object to plot on
    :param dat: dataframe containing information about associations
    :param ABC_type: it can be "ABC" for the mode, "mean" or "median"
    :param RR_conf: if True, include associations that are deemed significant by RR (not significant otherwise)
    :param a_conf: if True, include associations that are deemed significant by ABC (not significnat otherwise)
    :param colorbar: if True, include a color bar on the right side
    :param log: whether to plot in logarithmic scale
    :param xlim: limits of the x-axis
    :param ylim: limits of the y-axis
    :param title: title for the plot
    """
    data = dat
    if RR_conf is not None:
        data = data[data["fisher_sig"] == RR_conf]
    if a_conf is not None:
        data = data[data["a_sig"] == a_conf]
    print(f"{len(data)}/{len(dat)} data points")

    col = f"a+1(num, {ABC_type})" if ABC_type != "ABC" else ABC_type
    sns.scatterplot(data=data.sort_values(by="Pi"), x="RR", y=col, hue="Pi", ax=ax, palette="coolwarm")
    lim = [0.9 * min([data["RR"].min(), data[col].min()]), 1.1 * max([data["RR"].max(), data[col].max()])]
    ax.plot(lim, lim, linestyle="--", c="k")
    ax.axhline(1, linestyle=":", c="green"), ax.axvline(1, linestyle=":", c="green")

    norm = plt.Normalize(data['Pi'].min(), data['Pi'].max())
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    plt.colorbar(sm, label="Prevalence of condition, $P_i$") if colorbar else None
    ax.set(
        ylabel=f"Our association measure, $ABC_{{ij}}$"
               f"{((',' if a_conf else ', NOT') + ' significant') if a_conf is not None else ''}",
        xlabel=f"Relative Risk, $RR_{{ij}}$"
               f"{((',' if RR_conf else ', NOT') + ' significant') if RR_conf is not None else ''}")
    if log:
        ax.set(xscale="log", yscale="log")
        lim[0] = 0.9 if lim[0] > 1 else lim[0]
    else:
        lim[0] = -0.1
    ax.set(xlim=lim if xlim is None else xlim, ylim=lim if ylim is None else ylim,
           title=ABC_type if title is None else title)
    ax.grid()


def compute_communities(graph: nx.Graph, weight_attribute: str = None, negative_weights: bool = True,
                        **community_algorithm_kwargs):
    """
    Computes community detection of a (possibly weighted) networkx graph using the Clauset-Newman_Moore greedy
    modularity maximisation algorithm implemented by `networkx.algorithms.community.greedy_modularity_communities`.
    Resulting community labels are added to the graph nodes as the attributes 'modularity_class' and 'modularity_color'.

    :param graph: networkx.Graph
    :param weight_attribute: attribute of the graph edges used as weight in the community detection algorithm.
    :param negative_weights: whether negative weights are taken into account for community detection
    :return: community_dict: dictionary with node names as keys and community labels as values
    """
    filt_graph = graph.copy()
    if not negative_weights and weight_attribute is not None:
        filt_graph.remove_edges_from(
            [(u, v) for u, v, data in filt_graph.edges(data=True) if data[weight_attribute] < 1])

    communities = nx.algorithms.community.greedy_modularity_communities(filt_graph, weight=weight_attribute,
                                                                        **community_algorithm_kwargs)
    print("Modularity:", nx.community.modularity(filt_graph, communities, weight=weight_attribute),
          "coverage-performance", nx.community.partition_quality(filt_graph, communities))

    community_dict, community_color_dict = {}, {}
    for community_number, community in enumerate(communities):
        for name in community:
            community_dict[name] = community_number
            community_color_dict[name] = Spectral8[community_number]
    nx.set_node_attributes(graph, community_dict, 'modularity_class')
    nx.set_node_attributes(graph, community_color_dict, 'modularity_color')

    return community_dict


def build_network(multimorbidity_df: pd.DataFrame, var: str, cutoff=0, all_conditions: bool = False,
                  morbidity_names=None):
    """
    Build a network from an association dataframe.

    :param multimorbidity_df: dataframe with information about associations
    :param var: column of the dataframe to use for network edges
    :param cutoff: (float) cutoff value is imposed to filter out edges with low association values
    :param all_conditions: whether LTCs (nodes) without edges should be preserved
    :param morbidity_names:
    :return: network (nx.Graph), network_df (pd.DataFrame with the associations used in the network)
    """
    assert var in ("ABC", "RR", "Cij"), f"{var}"

    # Basic network data
    network_df = multimorbidity_df[multimorbidity_df["a_sig" if var == "ABC" else "fisher_sig"]] if var in (
        "ABC", "RR") else multimorbidity_df
    network_df = network_df.loc[:, ["i_abs", "j_abs", var, "i", "j", "Cij", "Pi", "Pj"]]

    # Filter network data by cutoff
    print(f"Original size: {len(network_df)} edges, "
          f"{len(pd.concat([network_df['i_abs'], network_df['j_abs']]).unique())} nodes", end=", ")
    network_df = network_df[np.abs(np.log(network_df[var])) > cutoff]
    print(f"filtered size: {len(network_df)} edges, "
          f"{len(pd.concat([network_df['i_abs'], network_df['j_abs']]).unique())} nodes.")

    # Build network
    network = nx.from_pandas_edgelist(network_df, source='i_abs', target='j_abs', edge_attr=[var])
    if all_conditions:
        network.add_nodes_from([i for i in range(len(morbidity_names)) if i not in list(network.nodes)])
    node_names = {i: morbidity_names[i] for i in list(network.nodes)}
    nx.set_node_attributes(network, name="name", values=node_names)

    return network, network_df


def plot_network(network: nx.Graph, network_df: pd.DataFrame, var: str, title: str = None, communities: bool = False,
                 community_weight: bool = True, layout=nx.spring_layout, figsize=(600, 600), interactive: bool = False,
                 prev_size: bool = False, span=8, beautify_name_func=lambda x, **kwargs: x):
    """
    Plot a network using Bokeh.

    For learning how to plot networks with bokeh:
    User guide: https://docs.bokeh.org/en/latest/docs/user_guide/topics/graph.html#network-graphs
    Tutorial: https://hub.gke2.mybinder.org/user/bokeh-bokeh-notebooks-g91zpr7c/doc/tree/tutorial/08%20-%20Graph%20and%20Network%20Plots.ipynb
    Maybe adding labels with this? https://docs.bokeh.org/en/latest/docs/user_guide/basic/annotations.html#labels
    https://stackoverflow.com/questions/47210530/adding-node-labels-to-bokeh-network-plots
    Tutorial: https://melaniewalsh.github.io/Intro-Cultural-Analytics/06-Network-Analysis/02-Making-Network-Viz-with-Bokeh.html
    Drag & drop: https://stackoverflow.com/questions/55785015/drag-drop-nodes-in-network-graph-with-bokeh
    But it doesn't seem to be properly implemented: https://stackoverflow.com/questions/62950074/drag-nodes-using-networkx-and-bokeh

    :param network: nx.Grapph network object to be visualised
    :param network_df: pd.DataFrame containing additional information about the associations used as nodes
    :param var: variable to use as edge weights
    :param title: title of the plot
    :param communities: whether community detection is applied and visualised
    :param community_weight: whether edge weights are regarded when performing community detection
    :param layout: layout to use for plotting the network
    :param figsize: figure size of the plot
    :param interactive: whether the plot allows for interactive visualisation using the mouse
    :param prev_size: whether LTC prevalence is used as node size
    :param span: range covered by the y-axis (note that we have set the x-axis as interactive but not the y-axis to
    allow to adjust for the optimal ratio)
    :param beautify_name_func: function to beautify label names. It needs to have a "sep" key, specifying the separator
    between words.
    :return:
    """
    assert var in ("ABC", "RR", "Cij"), f"{var}"
    assert layout in (nx.spring_layout, nx.circular_layout, nx.kamada_kawai_layout, nx.spiral_layout), layout

    network_df["log_weight"] = np.log(network_df[var])
    for _, row in network_df.iterrows():
        network[row['i_abs']][row['j_abs']].update(row[[var, "log_weight"]].to_dict())

    # Set node size
    node_size = 10 if not prev_size else {i: 50 * np.sqrt(
        network_df[network_df["i_abs"] == i]["Pi"].values[0] if i in network_df["i_abs"].values else
        (network_df[network_df["j_abs"] == i]["Pj"].values[0] if i in network_df["j_abs"].values else 0))
                                          for i in list(network.nodes)}
    nx.set_node_attributes(network, name="node_size", values=node_size)

    comm_dict = compute_communities(network,
                                    weight_attribute="log_weight" if community_weight else None) if communities else {}

    # Create figure
    hover_type = "nodes"
    HOVER_TOOLTIPS = [(var, f"@{var}"), ] if hover_type == "edges" else [("LTC", "@name"),
                                                                         ("Modularity Class", "@modularity_class"), ]
    figure_kwargs = dict(tooltips=HOVER_TOOLTIPS, tools="pan,wheel_zoom,save,reset,xwheel_zoom",
                         active_scroll='xwheel_zoom') if interactive else {}
    plot = figure(x_range=Range1d(-span, span), y_range=Range1d(-span, span),
                  title=var if title is None else title, width=figsize[0],
                  height=figsize[1], x_axis_location=None, y_axis_location=None, toolbar_location=None, **figure_kwargs)

    # Create network plot
    network_df["sign_colour"] = network_df[var].apply(lambda x: "red" if x < 1 else "black")
    edge_width_dict = {"ABC": 1.5, "RR": 1.5, "Cij": 20}  # line width of edges
    network_df["edge_line_width"] = edge_width_dict[var] * np.abs(network_df["log_weight"])
    network_df["layout_weight"] = edge_width_dict[var] * network_df["log_weight"]
    network_df.loc[network_df["layout_weight"] < 0, "layout_weight"] = 0  # no negative weights for layout algorithm
    [network[row['i_abs']][row['j_abs']].update(row[[var, 'sign_colour', 'edge_line_width', "layout_weight"]].to_dict())
     for _, row in network_df.iterrows()]

    layout_kwargs = dict(weight="layout_weight") if layout in (nx.spring_layout, nx.kamada_kawai_layout) else {}
    network_graph = from_networkx(network, scale=10, center=(0, 0), layout_function=layout, **layout_kwargs)

    # Nodes
    node_kwargs = dict(size="node_size", fill_color=(
        "skyblue" if not communities else "modularity_color"), )
    network_graph.node_renderer.glyph = Circle(**node_kwargs)
    network_graph.node_renderer.hover_glyph = Circle(line_width=2, **node_kwargs)
    network_graph.node_renderer.selection_glyph = Circle(line_width=2, **node_kwargs)

    # Edges
    edge_kwargs = dict(line_width="edge_line_width", line_color="sign_colour")
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.8, **edge_kwargs)
    network_graph.edge_renderer.selection_glyph = MultiLine(**edge_kwargs)
    network_graph.edge_renderer.hover_glyph = MultiLine(**edge_kwargs)

    # Highlighting Behaviour
    network_graph.selection_policy, network_graph.inspection_policy = [
        NodesAndLinkedEdges() if hover_type == "nodes" else EdgesAndLinkedNodes for _ in range(2)]

    plot.renderers.append(network_graph)

    # Add Labels
    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = [beautify_name_func(attrs["name"], sep=("\n" if layout == nx.circular_layout else " ")) for _, attrs
                   in list(network.nodes(data=True))],
    source = ColumnDataSource(dict(x=x, y=y, name=node_labels, **(
        {} if layout != nx.circular_layout else dict(x_offset=-20 + 3 * np.array(x), y_offset=-10 + 3 * np.array(y)))))
    labels_kwargs = {} if layout != nx.circular_layout else dict(x_offset="x_offset", y_offset="y_offset")
    labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color='white', text_font_size='10px',
                      background_fill_alpha=0.7, **labels_kwargs)
    plot.renderers.append(labels)

    plot.grid.grid_line_color = None

    return network, plot, [[network.nodes[i]["name"] for i in comm_dict if comm_dict[i] == community] for community
                           in np.unique(list(comm_dict.values()))]


if __name__ == '__main__':
    STAN_API = "cmdstanpy"
    if STAN_API == "pystan":
        import stan

        print("Compiling models with pystan version", stan.__version__)
    elif STAN_API == "cmdstanpy":
        import cmdstanpy

        cmdstanpy.show_versions()

    from lib.data import load_dataset
    from lib.model import ABCModel

    # M = 4e4
    dfo, names = load_dataset()  # , nrows=M)
    # mdel = train_age_group(8, dfo, num_warmup=50000, positive_levels=3, model_type=CooccurrenceModel)
    # mdel = train_age_group(9, dfo, num_chains=3, model_type=AtomicModel)

    # mdel = train_group(9, dfo, "Age", num_warmup=500, num_samples=20000, num_chains=1, model_type=AtomicModelHyperpriors)
    # mdel = train_group(9, dfo, "Age", num_warmup=500, num_samples=2000, num_chains=1, model_type=AtomicModelHyperpriors, model_name="MLTC_atomic_hyp_mult")
    # mdel = train_group(4, dfo, "Age85", num_warmup=500, num_samples=2000, num_chains=1, model_type=AtomicModelHyperpriors, model_name="MLTC_atomic_hyp_mult")
    # mdel = train_group(1, dfo, "Sex", num_warmup=500, num_samples=500, num_chains=3, model_type=AtomicModelHyperpriors)
    # mdel = train_group((9, 0), dfo, "Age-Sex", num_warmup=500, num_samples=20000, num_chains=1, model_type=AtomicModelHyperpriors)
    mdel = fit_model(dfo, "Age-Sex", (9, 0), num_warmup=1000, num_samples=1000, num_chains=10,
                     model_class=ABCModel, model_name="MLTC_atomic_hyp_mult")
