from datetime import datetime
import json
from pathlib import Path
import fire

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .misc import whi

exclude_list = [
        # "llm_openai",
        #"randomguess",
        ]

def main(
        paths: str = None,
        open_plot: bool = True,
        behavior: str = None,
        k: int = None,
        ):
    """
    Generate and display various plots and heatmaps for clustering results comparison.

    This function reads clustering results from a JSON file, processes the data,
    and creates several visualizations including:
    - A network graph showing relationships between different clustering methods
    - Heatmaps for F1 scores, Rand Index scores, and their comparisons against random guessing
    - Heatmaps for intrinsic metrics of clustering quality

    The function can either display the plots, save them to files, or both, depending on the parameters.

    Args:
        paths (str): Path to the directory containing the results JSON file.
        open_plot (bool): If True, display the plots. Default is True.
        behavior (str): Determines whether to 'return', 'store', or do 'both' with the generated plots.
        k (int): The number of clusters to use for the analysis.

    Returns:
        dict: A dictionary of plot figures if behavior is set to 'return' or 'both'.
    """
    if paths is None:
        raise Exception("You have to supply the path to the results")
    k = str(k)
    paths = str(paths)
    assert behavior in ["return", "store", "both"], "invalid behavior"

    res_file = Path(paths) / "pairwise.json"
    assert res_file.exists(), "file not found"

    with open(str(res_file), "r") as f:
        data = json.load(f)

    d = datetime.today()
    # time format is day/month/year
    today_date = f"{d.day:02d}/{d.month:02d}"
    ptitle = f"{paths.split('/')[-1]}, K={k} on {today_date} (n={data[0]['df_n_id']})"

    # create the network graph
    G = nx.Graph()
    weights = []
    for d in data:
        if exclude_if_match(d["path_to_res_A"], d["path_to_res_B"]):
            continue
        w = (d["perf_randindex_raw"][k] + d["perf_f1_raw"][k]) / 2
        weights.append(w)

    mea = sum(weights) / len(weights)
    mi = min(weights)
    colors = []
    for d in data:
        if exclude_if_match(d["path_to_res_A"], d["path_to_res_B"]):
            continue
        pa, pb = name_parser(d)
        pa += "\n\n"
        pb += "\n\n"
        assert pa != pb, f"duplicate results? '{pa}'"

        if pa not in G.nodes():
            if "FeaturesRaw" in pa or "FeaturesAgglo" in pa:
                colors.append("red")
            else:
                colors.append("black")
            G.add_node(pa)

        if pb not in G.nodes():
            if "FeaturesRaw" in pb or "FeaturesAgglo" in pb:
                colors.append("red")
            else:
                colors.append("black")
            G.add_node(pb)

        w = (d["perf_f1_mean"] + d["perf_randindex_mean"]) / 2
        adj_w = ((w-mi + 0.1) / (mea-mi + 0.1) + 2) ** 2

        G.add_edge(pa, pb, weight=adj_w, other=w)

    plt.figure()
    pos = nx.circular_layout(G)
    #pos = nx.spring_layout(G)
    widths = nx.get_edge_attributes(G, 'weight')
    orig_weight = nx.get_edge_attributes(G, 'other')
    def filt_node(nodename):
        # return True
        nodename = " ".join(nodename)
        if "raw" in nodename.lower():
            return True
        if "agg" in nodename.lower():
            return True
        return False
    weight_print = {(k): (str(v)[:5] if filt_node(k) else "") for k, v in orig_weight.items()}
    nodelist = G.nodes()
    nx.draw_networkx_nodes(G,
                           pos,
                           nodelist=nodelist,
                           node_size=150,
                           node_color=colors,
                           alpha=0.7)
    nx.draw_networkx_edges(G,
                           pos,
                           edgelist = widths.keys(),
                           width=list(widths.values()),
                           edge_color='lightblue',
                           alpha=0.2,
                           edge_cmap=plt.cm.Blues,
                           )
    nx.draw_networkx_edge_labels(G,
                                 pos,
                                 edge_labels=weight_print,
                                 font_color='black',
                                 alpha=0.7,
                                 font_size=5,
                                 rotate=True,
                                 label_pos=0.3,
                                 )
    nx.draw_networkx_labels(G,
                            pos=pos,
                            labels=dict(zip(nodelist, nodelist)),
                            font_color='black',
                            font_size=8,
                            clip_on=False,
                            )

    plt.box(False)
    plt.title(ptitle)
    graph_fig = plt.gcf()

    # create one confusion matrix for each k
    fontsize = 10

    run_name = []
    key_im = list(data[0]["intrinsic_metrics_names"])
    # the intrinsec metrics will be scaled between 0 and 1 so waiting for the highest value
    best_intrinsics = {k:0 for k in key_im}
    for d in data:
        for key in key_im:
            best_intrinsics[key] = max(best_intrinsics[key], d[f"{key}_A"][k])
        if exclude_if_match(d["path_to_res_A"], d["path_to_res_B"]):
            continue
        run_name.extend(name_parser(d))

    for key in key_im:
        if "silhouette" in key.lower():  # silhouette is -1 to 1, make it between 0 and 1
            best_intrinsics[key] += 1
            best_intrinsics[key] /= 2

    run_name = sorted(list(set(run_name)))
    run_name_wo_rand = [r for r in run_name if "randomguess" not in r.lower()]
    figsize = (len(run_name) + 1, len(run_name) + 1)

    # create empty df
    df_perf_f1 = pd.DataFrame(index=run_name, columns=run_name, data=np.nan)
    df_perf_randindex = pd.DataFrame(index=run_name, columns=run_name, data=np.nan)
    df_f1scores = pd.DataFrame(index=run_name, columns=run_name, data=np.nan)
    df_f1scores_vs_randint = pd.DataFrame(index=run_name, columns=run_name, data=np.nan)
    df_randindexscores = pd.DataFrame(index=run_name, columns=run_name, data=np.nan)
    df_randindexscores_vs_randint = pd.DataFrame(index=run_name, columns=run_name, data=np.nan)
    dfs_im = {
            key: pd.DataFrame(index=run_name_wo_rand, columns=run_name_wo_rand, data=np.nan)
            for key in key_im
            }
    dfs_cor_im = {
            key: pd.DataFrame(index=run_name_wo_rand, columns=run_name_wo_rand, data=np.nan)
            for key in key_im
            }

    for d in data:
        if exclude_if_match(d["path_to_res_A"], d["path_to_res_B"]):
            continue
        pa, pb = name_parser(d)
        assert np.isnan(df_perf_f1.loc[pa, pb]), "duplicate run?!"
        df_perf_f1.loc[pa, pb] = d["perf_f1_raw"][k]
        df_perf_randindex.loc[pa, pb] = d["perf_randindex_raw"][k]
        df_f1scores.loc[pa, pb] = d["f1"][k]
        df_randindexscores.loc[pa, pb] = d["randindex"][k]
        df_f1scores_vs_randint.loc[pa, pb] = d["f1_vs_randint"][k]
        df_randindexscores_vs_randint.loc[pa, pb] = d["randindex_vs_randint"][k]

        if "randomguess" not in pa.lower() and "randomguess" not in pb.lower():
            for key, df in dfs_im.items():
                vala = d[f"{key}_A"][k]
                valb = d[f"{key}_B"][k]

                if "silhouette" in key.lower():  # silhouette is -1 to 1, make it between 0 and 1, then scale it to the best
                    vala = (vala + 1) / 2 / best_intrinsics[key]
                    valb = (valb + 1) / 2 / best_intrinsics[key]
                elif "davies" in key.lower():
                    # scale from 0 to 1
                    vala /= best_intrinsics[key]
                    valb /= best_intrinsics[key]
                    # make is so that higher is better in the plot
                    vala = 1- vala
                    valb = 1- valb
                elif "calinski" in key.lower():
                    vala /= best_intrinsics[key]
                    valb /= best_intrinsics[key]
                else:
                    raise Exception(f"Unexpected intrinsic metric key: {key}")

                if vala + valb == 0:  # avoid division by zero just in case
                    adj = 0.00000001
                else:
                    adj = 0
                assert vala >= 0 and valb >= 0, f"{vala} or {valb} < 0"

                df.loc[pa, pb] = vala / (vala + valb + adj)
                df.loc[pb, pa] = valb / (vala + valb + adj)
                assert df.loc[pa, pb] >= 0 and df.loc[pa, pb] <= 1, f"{df.loc[pa, pb]}"
                assert df.loc[pb, pa] >= 0 and df.loc[pb, pa] <= 1, f"{df.loc[pb, pa]}"

                # store the correlation value
                dfs_cor_im[key].loc[pa, pb] = d[f"COR_{key}_A"][k]
                dfs_cor_im[key].loc[pb, pa] = d[f"COR_{key}_B"][k]

        # make symetrical
        for df in [
                df_perf_f1,
                df_perf_randindex,
                df_f1scores,
                df_f1scores_vs_randint,
                df_randindexscores,
                df_randindexscores_vs_randint,
                ]:
            df.loc[pb, pa] = df.loc[pa, pb]

    # make the intrinsic metrics go from 0 to 1
    for key, df in dfs_im.items():
        df.loc[:, :] -= np.nanmin(df.values.ravel())
        df.loc[:, :] /= np.nanmax(df.values.ravel())

    # create an additional intrinsic metric df that is the mean of the others
    df_all_im = pd.DataFrame(index=run_name_wo_rand, columns=run_name_wo_rand, data=0)
    df_all_im_cor = pd.DataFrame(index=run_name_wo_rand, columns=run_name_wo_rand, data=0)
    for key, df in dfs_im.items():
        df_all_im += df
        df_all_im_cor += dfs_cor_im[key]
    df_all_im /= len(dfs_im)
    df_all_im_cor /= len(dfs_cor_im)
    dfs_im["combined"] = df_all_im
    dfs_cor_im["combined"] = df_all_im_cor
    dfs_im["combined_diff"] = (df_all_im.abs() - df_all_im_cor.abs()).abs()
    dfs_cor_im["combined_diff"] = dfs_im["combined_diff"]

    # strip column and rown names
    df_perf_f1 = df_perf_f1.rename(columns=lambda x: x.strip())
    df_perf_f1 = df_perf_f1.rename(index=lambda x: x.strip())
    df_perf_randindex = df_perf_randindex.rename(columns=lambda x: x.strip())
    df_perf_randindex = df_perf_randindex.rename(index=lambda x: x.strip())
    df_f1scores = df_f1scores.rename(columns=lambda x: x.strip())
    df_f1scores = df_f1scores.rename(index=lambda x: x.strip())
    df_f1scores_vs_randint = df_f1scores_vs_randint.rename(columns=lambda x: x.strip())
    df_f1scores_vs_randint = df_f1scores_vs_randint.rename(index=lambda x: x.strip())
    df_randindexscores = df_randindexscores.rename(columns=lambda x: x.strip())
    df_randindexscores = df_randindexscores.rename(index=lambda x: x.strip())
    df_randindexscores_vs_randint = df_randindexscores_vs_randint.rename(columns=lambda x: x.strip())
    df_randindexscores_vs_randint = df_randindexscores_vs_randint.rename(index=lambda x: x.strip())

    random_thsh_f1 = np.nanmean(df_f1scores_vs_randint.values.ravel())
    if random_thsh_f1 == 0.5:  # avoid issues with vmin vmax
        random_thsh_f1 -= 0.001
    random_thsh_randindex = np.nanmean(df_randindexscores_vs_randint.values.ravel())
    if random_thsh_randindex == 0.5:  # avoid issues with vmin vmax
        random_thsh_randindex -= 0.001

    cmap = sns.dark_palette("#b285bc", n_colors=1000, as_cmap=True)
    # cmap.set_bad('white')  # Set color for cells with missing values
    # cmap.set_over('red')  # Set color for cells with missing values
    # cmap.set_under('yellow')  # Set color for cells with missing values

    hm_annot_kws = {"fontsize": "x-small"}

    # PERF F1 #############################################################
    plt.figure(figsize=figsize)
    sns.heatmap(
            df_perf_f1,
            annot=True,
            cmap=cmap,
            square=True,
            annot_kws=hm_annot_kws,
            cbar=False,
            vmin=min(random_thsh_f1, 1-random_thsh_f1),
            vmax=max(random_thsh_f1, 1-random_thsh_f1),
            )
    plt.xlabel('Runs', fontsize=fontsize) ; plt.ylabel('Runs', fontsize=fontsize)
    plt.xticks(rotation=30, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.box(False)
    plt.title(ptitle + f" (random={random_thsh_f1:2f})")
    plt.tight_layout()
    # Duplicate labels on top
    ax = plt.gca()
    ax_twin = ax.twiny()
    ax_twin.set_xlabel('Runs', fontsize=fontsize)
    ax_twin.tick_params(axis='x', rotation=30, labelsize=fontsize)
    ax_twin.set_xscale('linear')
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticks(ax.get_xticks())
    ax_twin.set_xticklabels(ax.get_xticklabels())
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    cm_perf_f1 = ax_twin.get_figure()

    # PERF RI #############################################################
    plt.figure(figsize=figsize)
    sns.heatmap(
            df_perf_randindex,
            annot=True,
            cmap=cmap,
            square=True,
            annot_kws=hm_annot_kws,
            cbar=False,
            vmin=min(random_thsh_randindex, 1-random_thsh_randindex),
            vmax=max(random_thsh_randindex, 1-random_thsh_randindex),
            )
    plt.xlabel('Runs', fontsize=fontsize) ; plt.ylabel('Runs', fontsize=fontsize)
    plt.xticks(rotation=30, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.box(False)
    plt.title(ptitle + f" (random={random_thsh_randindex:2f})")
    plt.tight_layout()
    # Duplicate labels on top
    ax = plt.gca()
    ax_twin = ax.twiny()
    ax_twin.set_xlabel('Runs', fontsize=fontsize)
    ax_twin.tick_params(axis='x', rotation=30, labelsize=fontsize)
    ax_twin.set_xscale('linear')
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticks(ax.get_xticks())
    ax_twin.set_xticklabels(ax.get_xticklabels())
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    cm_perf_randindex = ax_twin.get_figure()


    # SCORES F1 ###########################################################
    plt.figure(figsize=figsize)
    sns.heatmap(
            df_f1scores,
            annot=True,
            cmap=cmap,
            square=True,
            annot_kws=hm_annot_kws,
            cbar=False,
            vmin=0,
            vmax=1,
            )
    plt.xlabel('Runs', fontsize=fontsize) ; plt.ylabel('Runs', fontsize=fontsize)
    plt.xticks(rotation=30, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.box(False)
    plt.title(ptitle)
    plt.tight_layout()
    # Duplicate labels on top
    ax = plt.gca()
    ax_twin = ax.twiny()
    ax_twin.set_xlabel('Runs', fontsize=fontsize)
    ax_twin.tick_params(axis='x', rotation=30, labelsize=fontsize)
    ax_twin.set_xscale('linear')
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticks(ax.get_xticks())
    ax_twin.set_xticklabels(ax.get_xticklabels())
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    cm_f1 = ax_twin.get_figure()

    # SCORES RI ###########################################################
    plt.figure(figsize=figsize)
    sns.heatmap(
            df_randindexscores,
            annot=True,
            cmap=cmap,
            square=True,
            annot_kws=hm_annot_kws,
            cbar=False,
            vmin=0,
            vmax=1,
            )
    plt.xlabel('Runs', fontsize=fontsize) ; plt.ylabel('Runs', fontsize=fontsize)
    plt.xticks(rotation=30, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.box(False)
    plt.title(ptitle)
    plt.tight_layout()
    # Duplicate labels on top
    ax = plt.gca()
    ax_twin = ax.twiny()
    ax_twin.set_xlabel('Runs', fontsize=fontsize)
    ax_twin.tick_params(axis='x', rotation=30, labelsize=fontsize)
    ax_twin.set_xscale('linear')
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticks(ax.get_xticks())
    ax_twin.set_xticklabels(ax.get_xticklabels())
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    cm_randindex = ax_twin.get_figure()


    # F1 AGAINST RANDOMGUESS ##############################################
    plt.figure(figsize=figsize)
    sns.heatmap(
            df_f1scores_vs_randint,
            annot=True,
            cmap=cmap,
            square=True,
            annot_kws=hm_annot_kws,
            cbar=False,
            vmin=0,
            vmax=1,
            )
    plt.xlabel('Runs', fontsize=fontsize) ; plt.ylabel('Runs', fontsize=fontsize)
    plt.xticks(rotation=30, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.box(False)
    plt.title(ptitle)
    plt.tight_layout()
    # Duplicate labels on top
    ax = plt.gca()
    ax_twin = ax.twiny()
    ax_twin.set_xlabel('Runs', fontsize=fontsize)
    ax_twin.tick_params(axis='x', rotation=30, labelsize=fontsize)
    ax_twin.set_xscale('linear')
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticks(ax.get_xticks())
    ax_twin.set_xticklabels(ax.get_xticklabels())
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    cm_f1_vs_randint = ax_twin.get_figure()

    # RI AGAINST RANDOMGUESS ##############################################
    plt.figure(figsize=figsize)
    sns.heatmap(
            df_randindexscores_vs_randint,
            annot=True,
            cmap=cmap,
            square=True,
            annot_kws=hm_annot_kws,
            cbar=False,
            vmin=0,
            vmax=1,
            )
    plt.xlabel('Runs', fontsize=fontsize) ; plt.ylabel('Runs', fontsize=fontsize)
    plt.xticks(rotation=30, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.box(False)
    plt.title(ptitle)
    plt.tight_layout()
    # Duplicate labels on top
    ax = plt.gca()
    ax_twin = ax.twiny()
    ax_twin.set_xlabel('Runs', fontsize=fontsize)
    ax_twin.tick_params(axis='x', rotation=30, labelsize=fontsize)
    ax_twin.set_xscale('linear')
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticks(ax.get_xticks())
    ax_twin.set_xticklabels(ax.get_xticklabels())
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    cm_randindex_vs_randint = ax_twin.get_figure()

    # INTRINSIC METRICS ###################################################
    cms_im = {k: None for k in dfs_im.keys()}
    for key, df in dfs_im.items():
        plt.figure(figsize=figsize)
        sns.heatmap(
                df,
                annot=True,
                cmap=cmap,
                square=True,
                annot_kws=hm_annot_kws,
                cbar=False,
                vmin=0,
                vmax=1,
                )
        plt.xlabel('Runs', fontsize=fontsize) ; plt.ylabel('Runs', fontsize=fontsize)
        plt.xticks(rotation=30, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.box(False)
        plt.title(f"{key.title()} - " + ptitle)
        plt.tight_layout()
        # Duplicate labels on top
        ax = plt.gca()
        ax_twin = ax.twiny()
        ax_twin.set_xlabel('Runs', fontsize=fontsize)
        ax_twin.tick_params(axis='x', rotation=30, labelsize=fontsize)
        ax_twin.set_xscale('linear')
        ax_twin.set_xlim(ax.get_xlim())
        ax_twin.set_xticks(ax.get_xticks())
        ax_twin.set_xticklabels(ax.get_xticklabels())
        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
        cms_im[key] = ax_twin.get_figure()

        # correlation plot
        plt.figure(figsize=figsize)
        sns.heatmap(
                dfs_cor_im[key],
                annot=True,
                cmap=cmap,
                square=True,
                annot_kws=hm_annot_kws,
                cbar=False,
                vmin=0,
                vmax=1,
                )
        plt.xlabel('Runs', fontsize=fontsize) ; plt.ylabel('Runs', fontsize=fontsize)
        plt.xticks(rotation=30, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.box(False)
        plt.title(f"Correlation {key.title()} - " + ptitle)
        plt.tight_layout()
        # Duplicate labels on top
        ax = plt.gca()
        ax_twin = ax.twiny()
        ax_twin.set_xlabel('Runs', fontsize=fontsize)
        ax_twin.tick_params(axis='x', rotation=30, labelsize=fontsize)
        ax_twin.set_xscale('linear')
        ax_twin.set_xlim(ax.get_xlim())
        ax_twin.set_xticks(ax.get_xticks())
        ax_twin.set_xticklabels(ax.get_xticklabels())
        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
        cms_im[f"COR_{key}"] = ax_twin.get_figure()

    if behavior in ["both", "store"]:
        graph_fig.savefig(paths + "/network.png")
        cm_perf_f1.savefig(paths + "/confusion_matrix_perf_f1.png")
        cm_perf_randindex.savefig(paths + "/confusion_matrix_perf_randindex.png")
        cm_f1.savefig(paths + "/confusion_matrix_f1.png")
        cm_randindex.savefig(paths + "/confusion_matrix_randindex.png")
        cm_f1_vs_randint.savefig(paths + "/confusion_matrix_f1_vs_randint.png")
        cm_randindex_vs_randint.savefig(paths + "/confusion_matrix_randindex_vs_randint.png")
        for key, cm in cms_im.items():
            cm.savefig(paths + f"/confusion_matrix_{key}.png")

    if open_plot:
        graph_fig.show()
        cm_perf_f1.show()
        cm_perf_randindex.show()
        cm_f1.show()
        cm_randindex.show()
        cm_f1_vs_randint.show()
        cm_randindex_vs_randint.show()
        for key, cm in cms_im.items():
            cm.show()

    whi(f"Finished plots for k={k} : {paths}")

    if behavior in ["both", "return"]:
        out = {
                "Network": graph_fig,
                "F1 Performance (Score-Random)": cm_perf_f1,
                "Randindex Performance (Score-Random)": cm_perf_randindex,
                "F1": cm_f1,
                "RandIndex": cm_randindex,
                "F1 random": cm_f1_vs_randint,
                "Randindex random": cm_randindex_vs_randint,
                }
        for key, cm in cms_im.items():
            if key.startswith("COR_"):
                out[f"IM COR - {key.title()}"] = cm
            else:
                out[f"IM - {key.title()}"] = cm
        return out


def exclude_if_match(patha, pathb):
    for excl in exclude_list:
        if excl in patha.lower() or excl in pathb.lower():
            return True
    return False


def _np(norm, mode, cluster_method, dimred_method, n_dim):
    metric = "norm?"
    if "l1" in norm:
        metric = "L1"
    elif "l2" in norm:
        metric = "L2"

    method = "meth?"
    if "bisectingkmeans" in cluster_method:
        method = "BisKMeans"
    elif "kmeans" in cluster_method:
        method = "KMeans"
    elif "spectralcosine" in cluster_method:
        method = "SpectralCosine"

    dimred = "dimred?"
    if "pca" in dimred_method:
        dimred = "PCA"
    elif "umap" in dimred_method:
        dimred = "UMAP"
    elif "nmf" in dimred_method:
        dimred = "NMF"
    elif "dictionnarylearning" in dimred_method:
        dimred = "DicLear"
    elif "vae" in dimred_method:
        dimred = "VAE"

    # add a space to make them next to each other when sorted
    source = "mode?"
    if mode == "feat_raw":
        source = " FeaturesRaw"
    elif mode == "feat_agg":
        source = " FeaturesAgglo"
    elif mode == "feat_raw_no_norm":
        source = " FeaturesRawNoNorm"
    elif mode == "feat_agg_no_norm":
        source = " FeaturesAggloNoNorm"
    elif mode == "llm_openai":
        source = "OpenAI"
    elif mode.startswith("llm_"):
        source = _sbert_name_parser(mode)
    else:
        raise ValueError(f"uknown mode? {mode}")

    out = f"{source} {method} {metric} {dimred} D{n_dim}"
    return out


def _sbert_name_parser(mode):
    if mode == "llm_all-mpnet-base-v2".lower():
        return "AllMpnetBaseV2"
    elif mode == "llm_all-distilroberta-v1".lower():
        return "AllDistilRobertaV1"
    elif mode == "llm_distiluse-base-multilingual-cased-v2".lower():
        return "DistiluseBaseMultilCasedV2"
    elif mode == "llm_multi-qa-distilbert-cos-v1".lower():
        return "MultiQADistilbertCosV1"
    elif mode == "llm_paraphrase-multilingual-mpnet-base-v2".lower():
        return "ParaphMultilMpnetBaseV2"
    elif mode == "llm_msmarco-distilbert-cos-v5".lower():
        return "MSMarcoDistilbertCosV5"
    elif mode == "llm_clip-ViT-L-14".lower():
        return "ClipViTL14"
    elif mode == "llm_clip-ViT-B-32-multilingual-v1".lower():
        return "ClipViTB32MultiV1"
    elif mode == "llm_average_word_embeddings_glove.6B.300d".lower():
        return "Glove6B300d"
    elif mode == "llm_facebook-dpr-ctx_encoder-multiset-base".lower():
        return "DPRCtxMultiBase"
    elif mode == "llm_facebook-dpr-question_encoder-multiset-base".lower():
        return "DPRQuestMultiBase"
    elif mode == "llm_jina-embeddings-v2-base-en".lower():
        return "JinaV2BaseEn"
    elif mode.startswith("llm_random"):
        dim_n = mode.replace("llm_random_", "")
        return f" RandomVec{dim_n}D"
    else:
        return "sbert?"


def name_parser(res_content):
    # get A first
    if "randomGuess" == res_content["path_to_res_A"]:
        raise Exception("randomGuess is never supposed to be stored as path_to_res_A")
    runA = _np(
            norm=res_content["normA"].lower(),
            mode=res_content["modeA"].lower(),
            cluster_method=res_content["cluster_methodA"].lower(),
            dimred_method=res_content["dimred_methodA"].lower(),
            n_dim=res_content["dimensionA"],
            )
    if "randomGuess" == res_content["path_to_res_B"]:
        runB = "  RandomGuess"
    else:
        runB = _np(
                norm=res_content["normB"].lower(),
                mode=res_content["modeB"].lower(),
                cluster_method=res_content["cluster_methodB"].lower(),
                dimred_method=res_content["dimred_methodB"].lower(),
                n_dim=res_content["dimensionB"],
                )
    return [runA, runB]


if __name__ == "__main__":
    main = fire.Fire(main)
