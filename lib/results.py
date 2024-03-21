import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges, \
    LabelSet
from bokeh.palettes import Spectral8
from bokeh.plotting import figure, from_networkx

from lib.data import beautify_name, beautify_index
from lib.utils import identify_LTC


def get_top_associations(assoc_df: pd.DataFrame, columns=["a_f_sig_CI", "RR_f_sig_CI"], top_N_positive=10,
                         top_N_negative=None, **beautify_index_kwargs):
    """
    From an association dataframe, return the top 10 significant associations for ABC and RR and the negative
    associations for each of these measures.

    :param assoc_df: pandas Dataframe with association information.
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


def plot_tops(data, ax, panel_label=None, NS_pos=1, y_dict={}, negative=False, top_N_positive=10, top_N_negative=20):
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

        # print labels. The two lines below assume that all associations found by ABC are also found by RR. Otherwise, you will need to adjust the code
        ax.annotate(f'{label}, RR {row["RR_f_sig_CI"]}', (0, row["RR"]),
                    xytext=(-10, y_dict["RR"][label] if label in y_dict["RR"] else 0), ha='right',
                    color="black" if i < (top_N_positive if not negative else top_N_negative) else "grey",
                    **anno_kwargs)
        ax.annotate(f'ABC {row["a_f_sig_CI"]}, {label}', (1, row["ABC"]),
                    xytext=(10, y_dict["ABC"][label] if label in y_dict["ABC"] else 0), ha='left',
                    **anno_kwargs) if row["a_sig"] else None
        ax.annotate("ABC not significnat", (1, NS_pos), xytext=(10, 0), ha='left', **anno_kwargs)

        ax.set(yscale="log", ylim=[None, 1] if negative else None)
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


def plot_ABC_vs_RR(ax, dat, ABC_type="ABC", RR_conf=True, a_conf=True, colorbar=False, log=True,
                   xlim=None, ylim=None, fig=None, title=None):
    data = dat
    if RR_conf is not None:
        data = data[data["fisher_sig"] == RR_conf]
    if a_conf is not None:
        data = data[data["a_sig"] == a_conf]
    print(f"{len(data)}/{len(dat)} data points")

    col = f"a+1(num, {ABC_type})" if ABC_type != "ABC" else ABC_type
    sns.scatterplot(data=data.sort_values(by="Pi"), x="RR", y=col, hue="Pi", ax=ax, palette="coolwarm")
    # sns.relplot(data=dato[::-1], x="RR", y="a+1", hue="Pi", col="l", kind="scatter", ax=ax, palette="coolwarm");
    # sns.lineplot(data=dato, x="RR", y=f"a+1(num, {ABC_type})", marker='.', ax=ax, style="")
    lim = [0.9 * min([data["RR"].min(), data[col].min()]), 1.1 * max([data["RR"].max(), data[col].max()])]
    ax.plot(lim, lim, linestyle="--", c="k")
    ax.axhline(1, linestyle=":", c="green"), ax.axvline(1, linestyle=":", c="green")

    norm = plt.Normalize(data['Pi'].min(), data['Pi'].max())
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    fig.colorbar(sm, label="Prevalence of condition, $P_i$") if colorbar else None
    # ax.legend(title='$P_2$', loc='upper left');  # , labels=['Hell Yeh', 'Nah Bruh'])
    ax.set(
        ylabel=f"Our association measure, $ABC_{{ij}}${((',' if a_conf else ', NOT') + ' significant') if a_conf is not None else ''}",
        xlabel=f"Relative Risk, $RR_{{ij}}${((',' if RR_conf else ', NOT') + ' significant') if RR_conf is not None else ''}");
    if log:
        ax.set(xscale="log", yscale="log")
        if lim[0] > 1:
            lim[0] = 0.9
        # elif np.isclose(lim[0], 0):
        #     lim[0] = 1e-5
    else:
        lim[0] = -0.1
    ax.set(xlim=lim if xlim is None else xlim, ylim=lim if ylim is None else ylim,
           title=ABC_type if title is None else title)
    ax.grid()


def compute_communities(graph, weight, morbidity_names, var):
    filt_graph = graph.copy()
    if weight is None:
        filt_graph.remove_edges_from([(u, v) for u, v, data in filt_graph.edges(data=True) if data[var] < 1])

    communities = nx.algorithms.community.greedy_modularity_communities(filt_graph, weight=weight)
    print("Modularity:", nx.community.modularity(filt_graph, communities, weight=None), "coverage-performance",
          nx.community.partition_quality(filt_graph, communities))

    modularity_class, modularity_color = {}, {}
    comm_lists = []
    for community_number, community in enumerate(communities):
        comm_lists.append([beautify_name(morbidity_names[LTC], short=True) for LTC in community])
        for name in community:
            modularity_class[name] = community_number
            modularity_color[name] = Spectral8[community_number]
    nx.set_node_attributes(graph, modularity_class, 'modularity_class')
    nx.set_node_attributes(graph, modularity_color, 'modularity_color')
    return modularity_class, comm_lists


def build_network(multimorbidity_df, var, cutoff=0, all_conditions=False, morbidity_names=None):
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


def plot_network(network, network_df, var, title=None, communities=False, community_weight=True,
                 layout=nx.spring_layout, figsize=(600, 600), interactive=False, prev_size=False,
                 span=8, morbidity_names=None, beautify_list_func=lambda x: x):
    """
    For learning how to plot networks with bokeh:
    User guide: https://docs.bokeh.org/en/latest/docs/user_guide/topics/graph.html#network-graphs
    Tutorial: https://hub.gke2.mybinder.org/user/bokeh-bokeh-notebooks-g91zpr7c/doc/tree/tutorial/08%20-%20Graph%20and%20Network%20Plots.ipynb
    Maybe adding labels with this? https://docs.bokeh.org/en/latest/docs/user_guide/basic/annotations.html#labels
    https://stackoverflow.com/questions/47210530/adding-node-labels-to-bokeh-network-plots
    Tutorial: https://melaniewalsh.github.io/Intro-Cultural-Analytics/06-Network-Analysis/02-Making-Network-Viz-with-Bokeh.html
    Drag & drop: https://stackoverflow.com/questions/55785015/drag-drop-nodes-in-network-graph-with-bokeh
    But it doesn't seem to be properly implemented: https://stackoverflow.com/questions/62950074/drag-nodes-using-networkx-and-bokeh

    :param network:
    :param network_df:
    :param model:
    :param var:
    :param title:
    :param communities:
    :param community_weight:
    :param layout:
    :param figsize:
    :param show_plot:
    :param interactive:
    :param prev_size:
    :param span:
    :param morbidity_names:
    :param beautify_list_func:
    :return:
    """
    assert var in ("ABC", "RR", "Cij"), f"{var}"
    assert layout in (nx.spring_layout, nx.circular_layout, nx.kamada_kawai_layout, nx.spiral_layout), layout
    width_dict = {"ABC": 1.5, "RR": 1.5, "Cij": 20}  # width of edges
    signed_width_dict = {"ABC": 2, "RR": 2, "Cij": 20}  # width of edges (if signed)

    # Set node size
    # TODO: the code below is to implement size proportional to multimorbidity coupling, but needs fixing
    if False:  # var == "ABC" and not (layout == nx.circular_layout and communities == False) and not prev_size:
        couplings = model.get_multimorbidity_coupling()
        coupling_dict = {i: np.median(couplings[LTC_id]) for i, attrs in list(network.nodes(data=True)) if
                         (LTC_id := identify_LTC(LTC=attrs["name"], morb_cols=model.morb_names)[0]) is not None}
        nx.set_node_attributes(network, name="coupling", values=coupling_dict)
        node_size = {i: coupling * 1000 for i, coupling in coupling_dict.items()}
    else:
        node_size = 10 if not prev_size else {i: 50 * np.sqrt(
            network_df[network_df["i_abs"] == i]["Pi"].values[0] if i in network_df["i_abs"].values else
            (network_df[network_df["j_abs"] == i]["Pj"].values[0] if i in network_df["j_abs"].values else 0))
                                              for i in list(network.nodes)}
    nx.set_node_attributes(network, name="node_size", values=node_size)

    if communities:
        comm_dict, comm_lists = compute_communities(network, var=var, morbidity_names=morbidity_names,
                                                    weight=var if community_weight else None)

    if layout == nx.circular_layout:
        H = nx.Graph()
        for node in (model.LTC_df.sort_values(by="Counts", ascending=False)["i"].astype(
                int).values if not communities else comm_dict.keys()):
            H.add_node(node, **network.nodes(data=True)[node]) if node in network else None
        H.add_edges_from(network.edges(data=True))
        network = H

    # Create figure
    hover_type = "nodes"  # "edges"
    HOVER_TOOLTIPS = [(var, f"@{var}"), ] if hover_type == "edges" else [
        # ("LTC1", "@source"),  # when selection is for edges
        ("LTC", "@name"),  # ("Coupling", "@coupling")]
        ("Modularity Class", "@modularity_class"), ]  # when selection is for nodes
    figure_kwargs = {"tooltips": HOVER_TOOLTIPS,  # tooltips="index: @index"
                     "tools": "pan,wheel_zoom,save,reset,xwheel_zoom",
                     "active_scroll": 'xwheel_zoom'} if interactive else {}  # tools="hover",
    plot = figure(x_range=Range1d(-span, span), y_range=Range1d(-span, span),
                  title=f"{var}" if title is None else title, width=figsize[0],
                  height=figsize[1], x_axis_location=None, y_axis_location=None, toolbar_location=None,
                  **figure_kwargs)  # x_axis_location=None, y_axis_location=None)

    # Create network plot
    network_df["sign"] = network_df[var].apply(lambda x: "red" if x < 1 else "black")
    network_df["width"] = width_dict[var] * np.abs(np.log(network_df[var]))  # this is for plotting edge width
    network_df["log_edge"] = signed_width_dict[var] * np.log(network_df[var])  # this is for layout computation
    for _, row in network_df.iterrows():
        network[row['i_abs']][row['j_abs']].update(row[[var, 'sign', 'width', "log_edge"]].to_dict())

    layout_kwargs = {"weight": "log_edge"} if layout in (nx.spring_layout, nx.kamada_kawai_layout) else {}
    network_graph = from_networkx(network, scale=10, center=(0, 0), layout_function=layout, **layout_kwargs)

    # graph.node_renderer.data_source.data['index'] = list(range(len(ABC_G)))
    # graph.node_renderer.data_source.data['name'] = names_G
    # graph.node_renderer.data_source.data['colors'] = Category20_20

    # Nodes
    node_kwargs = dict(size="node_size", fill_color=(
        "skyblue" if not communities or layout == nx.circular_layout else "modularity_color"), )
    # line_color=("k" if not communities or layout == nx.circular_layout else "modularity_color"))
    network_graph.node_renderer.glyph = Circle(**node_kwargs)
    network_graph.node_renderer.hover_glyph = Circle(line_width=2, **node_kwargs)
    network_graph.node_renderer.selection_glyph = Circle(line_width=2, **node_kwargs)
    # network_graph.node_renderer.glyph.update(size=20, name="name")  # , fill_color="colors")

    # Edges
    edge_kwargs = {"line_width": "width", "line_color": "sign"}
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.8, **edge_kwargs)
    network_graph.edge_renderer.selection_glyph = MultiLine(**edge_kwargs)
    network_graph.edge_renderer.hover_glyph = MultiLine(**edge_kwargs)

    # Highlighting Behaviour
    network_graph.selection_policy, network_graph.inspection_policy = [
        NodesAndLinkedEdges() if hover_type == "nodes" else EdgesAndLinkedNodes for _ in range(2)]

    # plot.add_tools(PointDrawTool(renderers = [network_graph.node_renderer], empty_value = 'black'))
    plot.renderers.append(network_graph)

    # Add Labels
    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = beautify_list_func([attrs["name"] for _, attrs in list(network.nodes(data=True))],
                                     sep=("\n" if layout == nx.circular_layout else " "))  # list(network_graph.nodes())
    source = ColumnDataSource({'x': x, 'y': y, 'name': node_labels, **(
        {} if layout != nx.circular_layout else {"x_offset": -20 + 3 * np.array(x), "y_offset": -10 + 3 * np.array(
            y)})})  # [node_labels[i] for i in range(len(x))]})
    labels_kwargs = {} if layout != nx.circular_layout else {"x_offset": "x_offset", "y_offset": "y_offset"}  # -20}
    labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color='white', text_font_size='10px',
                      background_fill_alpha=0.7, **labels_kwargs)
    plot.renderers.append(labels)

    plot.grid.grid_line_color = None
    # save(plot, filename=f"{title}.html")

    return network, plot, comm_lists
