import time
import pdb
import os
from pathlib import Path
from sklearn.model_selection import GridSearchCV, ParameterGrid
from tqdm import tqdm
import random
from shutil import rmtree

from QuestEA import QuestEA

from torch.utils.tensorboard import SummaryWriter

import fire

from utils.compare_results import CompareResultsPairwise
from utils.plotter import main as Plotter
from utils.misc import IgnoreInGrid

print("Deleting cache folder")
rmtree("cache")

def do_grid_search(
        logdir="./tensorboard_runs",
        resultdir="./results_ignore_backups/",
        testing=False,
        debug=False,
        verbose=False,
        **kwargs,
        ):
    start = time.time()

    if kwargs:
        raise Exception(f"Unexpected parameter: '{kwargs}'")
    print("Trashing previous logdir")
    os.system(f"trash -i {logdir}")
    Path(logdir).mkdir(exist_ok=False)

    os.system(f"trash -i {resultdir}")
    Path(resultdir).mkdir(exist_ok=False)

    failed_file = Path(resultdir) / f"failed_{Path(resultdir).name}.txt"
    f = open(str(failed_file), "w")
    failed = []
    Path("logs.txt").touch()

    dataset_list = [
            # ordered by number of subjets
            "Hamilton",  # 98
            "AdolescentDepressionSRQA",  # 400
            "AdolescentDepressionCU",  # 400
            "RutledgeSmartphone",  # 1k
            "DASS",  # 6k
            "HEXACO",  # 12k
            "16PF",  # 20k
            "IPIP"  # 25k+
        ]
    param_grid = ParameterGrid({
            # "norm": ["l1", "l2"],
            "n_cluster": ["2-9"],
            "norm": ["l2"],
            "cluster_method": ["kmeans"], #"spectralcosine"],  # "bisectingkmeans"],  # kmeans, SpectralCosine
            "dimred_method": ["pca", "umap", "bvae", "dictionnarylearning"],  # pca/umap/nmf/bvae/dictionnarylearning
            "mode": [

                "llm_random_10",
                "llm_random_500",

                "feat_raw",
                "feat_agg",
                "feat_raw_no_norm",
                "feat_agg_no_norm",

                # sbert models
                "llm_all-mpnet-base-v2",
                "llm_clip-ViT-L-14",
                "llm_clip-ViT-B-32-multilingual-v1",
                "llm_all-distilroberta-v1",
                "llm_distiluse-base-multilingual-cased-v2",
                "llm_multi-qa-distilbert-cos-v1",
                "llm_paraphrase-multilingual-mpnet-base-v2",
                "llm_msmarco-distilbert-cos-v5",
                "llm_average_word_embeddings_glove.6B.300d",
                "llm_facebook-dpr-ctx_encoder-multiset-base",
                "llm_facebook-dpr-question_encoder-multiset-base",
                "llm_jina-embeddings-v2-base-en",

                "llm_openai",
                ],
            "n_components": [10],
            "sample_to_keep": [5_000],
            "testing": [testing],
            "result_dir": [resultdir],
            "verbose": [verbose],
            })


    # for each dataset, iterate over the whole parameter grid
    n = len(param_grid) * len(dataset_list)
    for datasetname in tqdm(dataset_list, desc="dataset", colour="magenta"):

        writer = SummaryWriter(logdir + f"/{datasetname}")

        param_count = 0
        for param in tqdm(param_grid, desc=datasetname, smoothing=0, colour="magenta"):
            param_count += 1
            tqdm.write(f"Parameters for {datasetname}: '{param}'")

            try:
                output = QuestEA(
                        datasetname=datasetname,
                        **param).output

                if output["intrinsic_metrics_plot"] is not None:
                    writer.add_figure(
                            tag="Intrinsic metrics plot",
                            figure=output["intrinsic_metrics_plot"],
                            global_step=param_count,
                            )

                predictions = output["predictions"]

                # store df print values as text
                writer.add_text(
                        tag=f"df_text {param['mode']} {param['norm']} {param['dimred_method']}",
                        text_string=output["self.df_text"],
                        )
                writer.add_text(
                        tag=f"df_answ {param['mode']} {param['norm']} {param['dimred_method']}",
                        text_string=output["self.df_answ"],
                        )

                # store embeddings for each subject
                metadata = []
                metadata_headers = ["subject_id", "mode", "norm",
                        "dimred_method"] + [f"pred K={k}" for k in output["n_cluster"]]
                for sid in list(set([l[0] for l in predictions.index.tolist()])):
                    metadata.append(
                            [
                               sid,
                               param["mode"],
                               param["norm"],
                               param["dimred_method"],
                               ] + [
                                   predictions.loc[ (sid, n_cluster), "prediction"]
                                   for n_cluster in output["n_cluster"]])
                writer.add_embedding(
                        mat=output["subjects_embeddings"].values,
                        metadata=metadata,
                        metadata_header=metadata_headers,
                        tag=f"EmbeddedSubjects {param['mode']} {param['norm']} {param['dimred_method']}",
                        )

                # and embeddings for the dataset
                if output["sentence_embeddings"] is not None:
                    metadata = []
                    metadata_headers = ["question_id", "mode", "norm",
                            "dimred_method"]
                    q_ids = output["sentence_embeddings"].index.tolist()
                    for qid in range(len(q_ids)):
                        metadata.append(
                                [
                                   q_ids[qid],
                                   param["mode"],
                                   param["norm"],
                                   param["dimred_method"],
                                   ])
                    writer.add_embedding(
                            mat=output["sentence_embeddings"].values,
                            metadata=metadata,
                            metadata_header=metadata_headers,
                            tag=f"UnansweredQuestionnaire {param['mode']} {param['norm']} {param['dimred_method']}",
                            )

            except IgnoreInGrid as err:
                tqdm.write(f"Error: '{err}'")
                f.write(f"{datasetname} {param}  -  {err}\n")
                failed.append(f"{datasetname} {param}  -  {err}")

                end = time.time() - start
                print(f"Total time so far: {end:2f}s")
                tqdm.write("Keep going despite error.")
            except Exception as err:
                tqdm.write(f"Error: '{err}'")
                f.write(f"{datasetname} {param}  -  {err}\n")
                failed.append(f"{datasetname} {param}  -  {err}")

                end = time.time() - start
                print(f"Total time so far: {end:2f}s")
                writer.flush()
                if debug:
                    pdb.post_mortem()
                else:
                    tqdm.write("Keep going despite error.")


        try:
            # after doing all the grid runs to one dataset, compare each run
            # of this dataset
            compared = CompareResultsPairwise(
                    paths=output["resultdir"],
                    behavior="both",
                    testing=testing,
                    verbose=param["verbose"],
                    )

            # for comp in compared:
            #     tag = comp["path_to_res_A"] + " vs " + comp["path_to_res_B"]
            #     tag = tag.replace("RES_", "")
            #     for it in comp["f1"].keys():
            #         writer.add_hparams(
            #                 {
            #                     "modeA": comp["modeA"],
            #                     "normA": comp["normA"],
            #                     "cluster_methodA": comp["cluster_methodA"],
            #                     "dimred_methodA": comp["dimred_methodA"],
            #                     "n_componentsA": comp["n_componentsA"],

            #                     "modeB": comp["modeB"],
            #                     "normB": comp["normB"],
            #                     "cluster_methodB": comp["cluster_methodB"],
            #                     "dimred_methodB": comp["dimred_methodB"],
            #                     "n_componentsB": comp["n_componentsB"],

            #                     "K": it,
            #                     },
            #                 {
            #                     "f1": comp["f1"][it],
            #                     "f1_vs_randint": comp["f1_vs_randint"][it],
            #                     "perf_f1_raw": comp["perf_f1_raw"][it],
            #                     "perf_f1_raw_2": comp["f1"][it] - comp["f1_vs_randint"][it],

            #                     "randindex": comp["randindex"][it],
            #                     "randindex_vs_randint": comp["randindex_vs_randint"][it],
            #                     "perf_randindex_raw": comp["perf_randindex_raw"][it],
            #                     "perf_randindex_raw_2": comp["randindex"][it] - comp["randindex_vs_randint"][it],
            #                     },
            #                 )
            #     writer.add_scalar("perf_f1_mean" + tag, comp["perf_f1_mean"])
            #     writer.add_scalar("perf_randindex_mean" + tag, comp["perf_randindex_mean"])

            for n_cluster in output["n_cluster"]:
                figs = Plotter(
                        paths=output["resultdir"],
                        open_plot=False,
                        behavior="both",
                        k=n_cluster,
                        )
                for k, v in figs.items():
                    writer.add_figure(
                            tag=k.replace("_", " ").title(),
                            figure=v,
                            global_step=n_cluster,
                            )

        except Exception as err:
            tqdm.write(f"Error: '{err}'")
            f.write(f"{datasetname} {param}  -  {err}\n")
            failed.append(f"{datasetname} {param}  -  {err}")

            end = time.time() - start
            print(f"Total time so far: {end:2f}s")
            if debug:
                pdb.post_mortem()
            else:
                tqdm.write("Keep going despite error.")
        writer.flush()
        writer.close()

    f.close()

    if not failed:
        print("No run failed.")
    else:
        print("Failed runs:\n" + "\n".join(failed))

    end = time.time() - start
    print(f"Total time: {end:2f}s")


if __name__ == "__main__":
    fire.Fire(do_grid_search)
