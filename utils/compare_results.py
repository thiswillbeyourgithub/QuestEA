import json
import pandas as pd
import copy
import random
from pathlib import Path
from tqdm import tqdm
import fire

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score
from joblib import Parallel, delayed

from .misc import whi


def CompareResultsPairwise(
        paths=None,
        behavior=None,
        testing=False,
        verbose=False,
        ):
    assert behavior in ["return", "store", "both"], "invalid behavior value"
    if paths is None:
        raise Exception("You have to supply the path to the results")
    orig_path = Path(paths)
    paths = [str(p) for p in Path(paths).rglob("RES_*.pickle")]
    assert len([p for p in paths if not Path(p).exists()]) == 0, "non existing files"
    paths.append("randomGuess")

    whi(f"Number of files found: '{len(paths)}'")
    iterator = []
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            assert i != j, "invalid iteration"
            iterator.append((i, j))

    results_para = Parallel(
            n_jobs=-1 if not testing else 1,
            backend="loky" if not testing else "threading",
            )(delayed(
                compareOnePair)(
                    pathA=paths[i],
                    pathB=paths[j],
                    testing_mode=testing,
                    verbose=verbose,
                    ) for i, j in tqdm(iterator, desc="pairwise"))

    scores_list = [r["scores_list"] for r in results_para]
    K_lists = [r["K_list"] for r in results_para]
    pathABs = [(r["path_to_res_A"], r["path_to_res_B"]) for r in results_para]

    assert len(scores_list) == len(K_lists), "invalid output length #2"
    assert len(K_lists) == len(pathABs), "invalid output length #3"

    results = []
    for inc, (i, j) in enumerate(tqdm(iterator, desc="collating results")):
        temp = {
                "path_to_res_A": pathABs[inc][0].split("/")[-1],
                "path_to_res_B": pathABs[inc][1].split("/")[-1],
                "df_n_id": results_para[inc]["df_shape_nid"],
                "intrinsic_metrics_names": results_para[inc]["intrinsic_metrics_names"],
                }
        for scorer in ["f1", "randindex"]:
            assert len(scores_list[inc]) == len(K_lists[inc])
            perf_raw = [(x[scorer]-y[f"{scorer}_vs_randint"]) for x, y in zip(scores_list[inc], scores_list[inc])]
            perf_max = max(perf_raw)
            perf_max_k = K_lists[0][perf_raw.index(perf_max)]
            perf_mean = mean(perf_raw)

            temp[f"perf_{scorer}_max"] = perf_max
            temp[f"perf_{scorer}_max_k"] = perf_max_k
            temp[f"perf_{scorer}_mean"] = perf_mean
            temp[f"perf_{scorer}_raw"] = {k: f for k, f in zip(K_lists[inc], perf_raw)}
            temp[scorer] = {k: f[scorer] for k, f in zip(K_lists[inc], scores_list[inc])}
            temp[f"{scorer}_vs_randint"] = {k: f[f"{scorer}_vs_randint"] for k, f in zip(K_lists[inc], scores_list[inc])}


        for word in ["mode", "norm", "cluster_method", "dimred_method", "n_components"]:
            temp[word + "A"] = results_para[inc][word + "A"]
            temp[word + "B"] = results_para[inc][word + "B"]
        for m_name in results_para[inc]["intrinsic_metrics_names"]:
            temp[f"{m_name}_A"] = {k: float(f) for k, f in zip(K_lists[inc], json.loads(results_para[inc][f"{m_name}_A"]))}
            temp[f"{m_name}_B"] = {k: float(f) for k, f in zip(K_lists[inc], json.loads(results_para[inc][f"{m_name}_B"]))}
            temp[f"COR_{m_name}_A"] = {k: float(f) for k, f in zip(K_lists[inc], json.loads(results_para[inc][f"COR_{m_name}_A"]))}
            temp[f"COR_{m_name}_B"] = {k: float(f) for k, f in zip(K_lists[inc], json.loads(results_para[inc][f"COR_{m_name}_B"]))}
        temp["dimensionA"] = results_para[inc]["df_shape_ndimA"]
        temp["dimensionB"] = results_para[inc]["df_shape_ndimB"]

        for k, v in temp.items():
            if isinstance(temp[k], np.int64):
                temp[k] = int(v)
            elif isinstance(temp[k], np.float64):
                temp[k] = float(v)
        results.append(temp)

    if behavior in ["both", "store"]:
        with open(str(orig_path / "pairwise.json"), "w") as f:
            results2 = sorted(
                    results,
                    key=lambda x: x[f"perf_{scorer}_mean"],
                    reverse=True,
                    )
            json.dump(results2, f, indent=2)
    if behavior in ["both", "return"]:
        return results2


def compareOnePair(pathA, pathB, testing_mode, verbose):
    if verbose:
        tqdm.write(f"\n\n######\n* {pathA.split('/')[-1]}\n* {pathB.split('/')[-1]}")
    assert pathA != "randomGuess", "invalid randomguess in pathA"

    df_A = pd.read_pickle(pathA)
    predA = pd.read_pickle(pathA.replace("RES_", "PRED_", 1))

    if pathB == "randomGuess":
        df_B = copy.deepcopy(df_A)
        predB = copy.deepcopy(predA)
    else:
        df_B = pd.read_pickle(pathB)
        predB = pd.read_pickle(pathB.replace("RES_", "PRED_", 1))

    if pathB == "randomGuess":
        nmax = max(predB.values.ravel())
        nmin = min(predB.values.ravel())
        for row in predB.itertuples():
            sub = row[0][0]
            k = row[0][1]
            predB.loc[(sub, k), "prediction"] = random.randint(nmin, nmax)
        assert max(predB.values.ravel()) == nmax and min(predB.values.ravel()) == nmin, (
            "error in randomization")
    else:
        assert not df_A.equals(df_B) and not predA.equals(predB), "A and B are the same!"
    assert df_A.index.equals(df_B.index), "A and B don't share the same n_cluster!"
    assert df_A["intrinsic_metrics_names"].equals(df_B["intrinsic_metrics_names"]), "A and B don't share intrinsic_metrics_names!"
    assert df_A["df_shape_nid"].equals(df_B["df_shape_nid"]), "A and B don't share df_shape_nid!"

    # check if the subject id of the prediction differ
    sa = set([li[0] for li in predA.index.tolist()])
    sb = set([li[0] for li in predB.index.tolist()])
    diff = sorted(list(sa ^ sb))
    if diff:
        common_ids = sa & sb
        total = (len(sa) + len(sb)) // 2
        prct = len(diff) / total * 100
        assert not predA.index.equals(predB.index), "unexpected error"
        if testing_mode:
            tqdm.write(
                    f"difference in index: '{diff}' out of {total} ({int(prct)}%. Ignoring because testing mode is on.")
        else:
            raise Exception(
                f"difference in index: '{diff}' out of {total} ({int(prct)}%. Stopping.")
    else:
        common_ids = sa

    # gather the scores after hungarian method
    K_list = df_A["n_cluster"].tolist()
    scores_list = []
    for k in K_list:
        pred1 = np.array([int(predA.loc[(cid, k), "prediction"]) for cid in common_ids])
        pred2 = np.array([int(predB.loc[(cid, k), "prediction"]) for cid in common_ids])

        scores, pred1, pred2 = remap_label_then_score(pred1, pred2, verbose)
        scores_list.append(scores)

        if verbose:
            tqdm.write(f"\nComparing K={k}")
            tqdm.write(f"            Done with K={k}: scores={scores}\n")

    out = {
            "df_shape_nid": df_B.iloc[0]["df_shape_nid"],
            "K_list": K_list,
            "scores_list": scores_list,
            "intrinsic_metrics_names": json.loads(df_A.iloc[0]["intrinsic_metrics_names"]),

            "path_to_res_A": pathA,
            "df_shape_ndimA": df_A.iloc[0]["df_shape_ndim"],
            "modeA": df_A.iloc[0]["mode"],
            "normA": df_A.iloc[0]["norm"],
            "cluster_methodA": df_A.iloc[0]["cluster_method"],
            "dimred_methodA": df_A.iloc[0]["dimred_method"],
            "n_componentsA": df_A.iloc[0]["n_components"],

            "path_to_res_B": pathB,
            "df_shape_ndimB": df_B.iloc[0]["df_shape_ndim"],
            "modeB": df_B.iloc[0]["mode"],
            "normB": df_B.iloc[0]["norm"],
            "cluster_methodB": df_B.iloc[0]["cluster_method"],
            "dimred_methodB": df_B.iloc[0]["dimred_method"],
            "n_componentsB": df_B.iloc[0]["n_components"],
            }

    # add the intrinsic metrics
    for m_name in json.loads(df_A.iloc[0]["intrinsic_metrics_names"]):
        out[f"{m_name}_A"] = json.dumps(df_A[m_name].apply(lambda x: float(x)).tolist())
        out[f"{m_name}_B"] = json.dumps(df_B[m_name].apply(lambda x: float(x)).tolist())
        out[f"COR_{m_name}_A"] = json.dumps(df_A[f"COR_{m_name}"].apply(lambda x: float(x)).tolist())
        out[f"COR_{m_name}_B"] = json.dumps(df_B[f"COR_{m_name}"].apply(lambda x: float(x)).tolist())

    return out

def score_func(pred1, pred2):
    return {
            "randindex": adjusted_rand_score(pred1, pred2),
            "f1":  (f1_score(pred1, pred2, average="micro") + f1_score(pred2, pred1, average="micro")) / 2,
            }

def remap_label_then_score(pred1, pred2, verbose):
    """ 
    Hungarian method to remap labels and then score the predictions.

    This remapping is necessary because clustering algorithms may assign 
    different numerical labels to the same conceptual clusters across different 
    runs or methods. For example, what one method calls cluster '1' might be 
    essentially the same as what another method calls cluster '3'. 

    The Hungarian method finds the optimal one-to-one mapping between the 
    label sets, allowing for a fair comparison of the clustering results 
    regardless of the arbitrary numerical labels assigned.
    """

    old_pred1 = pred1.copy()
    old_scores = score_func(old_pred1, pred2)

    # remap labels
    max_value = np.max((np.max(pred1), np.max(pred2)))
    counts = np.zeros((max_value + 1, max_value + 1), dtype=int)
    for lab1, lab2 in zip(pred1, pred2):
        counts[lab1, lab2] += 1
    indices = linear_sum_assignment(counts.max() - counts, maximize=False)
    new_map = dict((key, val) for key, val in zip(*indices))

    pred1 = np.vectorize(new_map.get)(pred1)
    scores = score_func(pred1, pred2)

    # only keep best accuracy
    assert not [k for k in old_scores.keys() if scores[k] < old_scores[k]], (
        f"lower accuracy after remapping labels: scores={scores} old_scores={old_scores}")

    # return the score against random integers
    nmax = max(pred1)
    nmin = min(pred1)
    n = 10
    for k in list(scores.keys()):
        scores[f"{k}_vs_randint"] = 0
    for _ in range(n):
        temp = score_func(
                pred1,
                [random.randint(nmin, nmax) for v in pred1],
                )
        for k in temp:
            scores[f"{k}_vs_randint"] += temp[k]
    for k in temp:
        scores[f"{k}_vs_randint"] /= n

    return scores, pred1, pred2


def mean(li):
    "easier to read"
    return sum(li)/len(li)


if __name__ == "__main__":
    main = fire.Fire(CompareResultsPairwise)
