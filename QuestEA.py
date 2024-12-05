import logging
import pickle
import psutil
from datetime import datetime
import time
import gc
import json
import os
from pathlib import Path
import fire

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import umap
from sklearn.decomposition import PCA, NMF, DictionaryLearning
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, SpectralClustering, BisectingKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


from joblib import Parallel, delayed, Memory

from utils.dataloader import Dataloader
from utils.dynamic_batch import DynamicBatchIterator
from utils.misc import (set_global_logging_level, whi, yel, red,
                        _get_sentence_encoder,
                        IgnoreInGrid
                        )
from utils.bvae.bvae import OptimizedBVAE

# patch sklearn makes it faster
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except Exception as err:
    print(f"Exception when patching sklearn: '{err}'")


# reduce logging verbosity
set_global_logging_level(logging.ERROR,
        ["sklearn", "sklearnx", "transformers", "nlp", "torch", "tensorflow", "nltk"])

dataloader_cache = Memory("cache/dataloader_cache", verbose=0)

@dataloader_cache.cache(ignore=["mode"])
def cached_dataloader(datasetname, mode, mode_type, norm_f):
    "short wrapper to use joblib memory for caching the datasets"
    red(f"Loading datasets instead of using cache: {datasetname} {mode} {norm_f}")
    dl = Dataloader(
            datasetname=datasetname,
            mode=mode,
            norm_f=norm_f)
    return dl.df_answ, dl.df_text

class QuestEA:
    def __init__(
            self,
            mode=None,
            result_dir="./results_ignore_backups",
            note="note",
            sample_to_keep=None,
            datasetname=None,
            n_components=None,
            norm="l2",
            testing=False,
            skip_plot=False,
            show_plot=False,
            n_cpus=-1,
            cluster_method="kmeans",
            n_cluster="2-8",
            dimred_method="pca",
            skip_embeddings_cache=False,
            verbose=False,
            *args,
            **kwargs,
            ):
        """
        Parameters
        ----------
        mode : str
            Either feat_raw, feat_agg or feat_raw_no_norm or feat_agg_no_norm
            or starts with llm_

            * 'feat' stands for 'features' and means that the input of the
              clustering method will be straight from the user answer to the
              inventory. Either directly take the normalized outputs as features
              or use a post processing. For example the 5 factor is made of
              40 questions, so feat_raw means 40 dimensions where each dimension
              is the answer to a question and feat_agg means 5 dimensions where
              each dimension is one of the computed factor according to the
              inventory designer's formula.
            * 'llm_' need to be either 'llm_openai' or be followed by
              the name from a pretrained model at
              https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/
              For example: 'llm_clip-ViT-B-32' (case sensitive)
              OR can be 'llm_random' which is used for testing. It returns
              random vectors.
        result_dir : str, optional
            Path to where the raw results will be stored. This dir makes
            the link between QuestEA and the utils scripts to compare and
            plot. Default is './results_ignore_backups/'.
        note : str
            String to annotate results.
        sample_to_keep : int, optional
            Number of samples to keep from the dataset. If None, will keep all the data.
            If testing is True, will keep even less.
        datasetname : str
            Name of the dataset to load.
        n_components : int, optional
            If int, number of dimensions to keep with the dimred method. Default is None.
        norm : str, optional
            Either l1 or l2 norm. Passed to the dataloader and applied at
            several key steps. Default is "l2".
        testing : bool, optional
            If True, only try on 1000 points. Default is False.
        skip_plot : bool, optional
            If False, will create and save plot. Default is True.
        show_plot : bool, optional
            If True, will automatically open the plot instead of just saving it. Default is False.
        n_cpus : int, optional
            Used when using batch processing to compute patient embedding
            based on the answer to the inventory. Default is -1.
        cluster_method : str, optional
            Kmeans, bisectingkmeans or spectralcosine.
            Lowercase is applied. Default is 'kmeans'.
        n_cluster : str, optional
            If '2-8' then will look for cluster for k in range(2, 9). Default is '2-8'.
        dimred_method : str, optional
            Can be pca, umap, nmf, dictionnarylearning, bvae. Default is 'pca'.
        skip_embeddings_cache : bool, optional
            If True, will recompute all embeddings instead of computing them. This
            can imply cost if you are not self hosting an embedding model. Default is False.
        verbose : bool, optional
            Increase verbosity. Default is False.
        """
        # argument validity checking
        if args:
            raise SystemExit(f"Unexpected args: {args}")
        if kwargs:
            raise SystemExit(f"Unexpected kwargs: {kwargs}")
        if datasetname is None:
            raise Exception("you have to supply datasetname")

        assert norm in ["l1", "l2"], "invalid norm argument"
        cluster_method == cluster_method.lower()
        assert cluster_method in ["kmeans", "bisectingkmeans", "spectralcosine"], "invalid cluster_method"
        assert str(n_cpus).isdigit() or str(n_cpus)[1:].isdigit(), "invalid type for n_cpus"
        assert mode in [
                "feat_raw",
                "feat_agg",
                "feat_raw_no_norm",
                "feat_agg_no_norm",
                ] or mode.startswith("llm_"), "you have to specify a mode"
        assert isinstance(skip_embeddings_cache, bool), "skip_embeddings_cache has to be a boolean"
        assert dimred_method in ["pca", "umap", "nmf", "dictionnarylearning", "bvae"], "invalid dimred method"

        # store as attribute
        self.datasetname = datasetname
        self.mode = mode
        self.norm = norm
        self.sample_to_keep = sample_to_keep
        self.testing = testing
        self.note = note
        self.n_components = n_components
        self.show_plot = show_plot
        self.skip_plot = skip_plot
        self.skip_embeddings_cache = skip_embeddings_cache
        self.n_cpus = int(n_cpus) if not testing else 1
        n_c = n_cluster.split("-")
        self.n_cluster = list(range(int(n_c[0]), int(n_c[1]) + 1))
        self.cluster_method = cluster_method
        self.dimred_method = dimred_method
        self.verbose = verbose

        # init vars
        self.resultdir = Path(result_dir)
        self.resultdir.mkdir(exist_ok=True)
        self.resultdir = self.resultdir / datasetname
        self.resultdir.mkdir(exist_ok=True)
        self.batch_dir = self.resultdir / "batch_cache"
        self.norm_f = Normalizer(norm=self.norm, copy=True)
        self.note += f"_{cluster_method}"
        self.note += f"_{datasetname}"
        d = datetime.today()
        self.timestamp = f"{d.day}{d.month}{d.year}{int(time.time())}"

        # used to return the sentence embeddings to tensorboard
        self.sentence_embeddings = None

        self._load_dataset()
        assert self.df_answ.values.dtype == "float32", f"Invalid df_answ dtype: {self.df_answ.values.dtype}"

        if self.mode.startswith("llm_"):
            self._embed_inventory()
        else:
            assert self.mode in ["feat_raw", "feat_agg",
                                 "feat_raw_no_norm", "feat_agg_no_norm"], (
                f"Unexpected mode: '{self.mode}'")
            # despite the name df_EmbedInventories will not contain
            # embeddings but the raw answers
            self.df_EmbedInventories = self.df_answ

        self._clustering()

        whi("End of run")

    def _load_dataset(self):
        self.df_answ, self.df_text = cached_dataloader(
            datasetname=self.datasetname,
            mode=self.mode,
            norm_f=self.norm_f,
            mode_type="llm_" if self.mode.startswith("llm_") else self.mode,
            )

        if self.sample_to_keep and self.sample_to_keep < len(self.df_answ):
            whi(f"sample_to_keep values: '{self.sample_to_keep}")
            whi(f"Keeping only {self.sample_to_keep} samples.")
            whi(f"Previous shape: {self.df_answ.shape}")
            self.df_answ = self.df_answ.sample(n=self.sample_to_keep, replace=False, random_state=42)
            whi(f"New shape: {self.df_answ.shape}")
        else:
            whi("argument sample_to_keep here means keeping all the data")

        assert len(self.df_answ.index.tolist()), "empty df_answ"

    def _embed_inventory(self):
        # if testing mode: keep only a tiny subset
        n = 1000
        if self.testing and n < len(self.df_answ):
            whi(f"\nKeeping only {n} data points for testing")
            self.df_answ = self.df_answ.sample(n, random_state=42)
            self.note += "_testing"

        # if the mode is an embedding mode, compute the embeddings:
        # use the raw answer as input for the embeddings then normalize
        assert self.mode.startswith("llm_"), (
            f"invalid mode: '{self.mode}'")
        self.df_answ = self.df_answ.loc[:, self.df_text.index].astype(np.float32)

        sentence_encoder = _get_sentence_encoder(
                mode=self.mode,
                cache=Memory(
                    "cache",
                    verbose=self.verbose,
                    ) if not self.skip_embeddings_cache else None,
                normalizer=self.norm_f,
                )

        # create a dummy vector to be sure of the shape of the embedding
        if self.verbose:
            whi("creating dummy vector")
        test_vec = sentence_encoder(
                sentences=["I love cats"],
                )
        assert test_vec.shape[0] == 1

        # encode vector of each sentence
        whi("\n\nEncoding sentences...")
        vec_list = [f"V_{i + 1:04d}" for i in range(max(test_vec.shape))]
        df_EmbSent = pd.DataFrame(
                index=self.df_text.index.tolist(),
                columns=vec_list,
                dtype=np.float32,
                )
        vecs = sentence_encoder(sentences=self.df_text["sentence"].tolist())
        assert vecs.shape[1] == len(vec_list), "Unexpected vector dimensions"
        assert vecs.shape[0] == len(df_EmbSent.index), "Unexpected number of vectors"
        assert not (vecs[0, :] == vecs[-1, :]).all(), "Vectors seem identical!"
        df_EmbSent.loc[:, vec_list] = vecs
        assert df_EmbSent.values.max() != df_EmbSent.values.min(), (
                "min and max of df_EmbSent are the same!")

        # multiprocessed batches
        def embed_batch_of_inventory(batch, fid, norm_f, save=True):
            """multiprocessed batch that takes the answer to the inventory
            of a batch of user, multiply this array by the embeddings of the
            sentences, then normalize so that the sum for each user is 1
            """
            answers = batch.loc[:, self.df_text.index].values
            assert answers.ravel().min() > 0, "answers can't contain 0"
            assert np.isclose((answers % 1).sum(), 0), "answers to inventory is supposed to be integers"
            # score = np.sum(answers[:, :, np.newaxis] * df_EmbSent.values[np.newaxis, :, :], axis=1)
            score = np.max(answers[:, :, np.newaxis] * df_EmbSent.values[np.newaxis, :, :], axis=1)
            score = norm_f.fit_transform(score + abs(score.min())).astype(np.float32)
            if norm_f.norm.lower() == "l1":
                assert np.isclose(np.abs(score).sum(axis=1).mean(), 1), (
                    "Failed L1 normalization")
            elif norm_f.norm.lower() == "l2":
                assert np.isclose(np.square(score).sum(axis=1).mean(), 1), (
                    "Failed L2 normalization")
            else:
                raise ValueError(f"Unexpected normalization: {norm_f.norm}")
            if save:
                fid = f"{fid:05d}"
                with open(str(self.batch_dir / f"{fid}_score.json"), "wb") as f:
                    pickle.dump(score, f)
                with open(str(self.batch_dir / f"{fid}_index.json"), "wb") as f:
                    pickle.dump(batch.index.values, f)
                return fid
            else:
                return score, batch.index.values

        self.batch_dir.mkdir(exist_ok=True)
        # make sure the batch dir is empty
        [f.unlink() for f in self.batch_dir.rglob("*")]

        whi("Create new batch")
        if self.n_cpus != -1:
            p = psutil.Process()
            p.cpu_affinity([x for x in range(self.n_cpus)])
        batches = DynamicBatchIterator(
                df=self.df_answ,
                crop_df=True,
                n_cpus=self.n_cpus,
                )
        results = Parallel(
                n_jobs=self.n_cpus,
                backend="loky" if not self.testing else "threading",
                )(delayed(
            embed_batch_of_inventory)(
                batch=batch,
                fid=i,
                norm_f=self.norm_f
                ) for i, batch in enumerate(batches))
        if self.n_cpus != -1:
            # reset cpu affinity
            p = psutil.Process()
            p.cpu_affinity([])

        scores = []
        indices = []
        scores_buff = []
        indices_buff = []
        buff_limit = 1000
        for i, fid in enumerate(tqdm(results, desc="loading results from files", smoothing=0.0)):
            with open(str(self.batch_dir / f"{fid}_score.json"), "rb") as f:
                scores_buff.append(pickle.load(f).astype(np.float32))
                if len(scores_buff) > buff_limit or fid == results[-1]:
                    if len(scores):
                        scores = np.concatenate((scores, *scores_buff))
                    else:
                        scores = np.concatenate((scores_buff))
                    scores_buff = []
            with open(str(self.batch_dir / f"{fid}_index.json"), "rb") as f:
                indices_buff.append(pickle.load(f))
                if len(indices_buff) > buff_limit or fid == results[-1]:
                    if len(indices):
                        indices = np.concatenate((indices, *indices_buff))
                    else:
                        indices = np.concatenate((indices_buff))
                    indices_buff = []

        df_EmbedInventories = pd.DataFrame(
                scores,
                index=indices,
                columns=df_EmbSent.columns,
                dtype=np.float32,
                )

        self.df_EmbedInventories = df_EmbedInventories
        self.sentence_embeddings = df_EmbSent


    def _clustering(self):
        """ perform dimension reduction then clustering over the
        embedded inventories then plot intrinsic metrics then return
        """
        # check that the data is normalized
        if "_no_norm" not in self.mode:
            if self.norm_f.norm.lower() == "l1":
                sums = self.df_EmbedInventories.abs().sum(axis=1)
                assert np.isclose(sums.max(), 1), "EmbedInventories is not L1 normalized"
                assert np.isclose(sums.min(), 1), "EmbedInventories is not L1 normalized"
            elif self.norm_f.norm.lower() == "l2":
                sums = (self.df_EmbedInventories ** 2).sum(axis=1)
                assert np.isclose(sums.max(), 1), "EmbedInventories is not L2 normalized"
                assert np.isclose(sums.min(), 1), "EmbedInventories is not L2 normalized"
            else:
                raise ValueError(self.norm)
        else:
            whi(f"No norm mode for {self.mode}, don't check if data was normalized")

        # depending on the dimension of the df, apply PCA or not
        index = self.df_EmbedInventories.index.tolist()
        whi(f"Original df_EmbedInventories shape: {self.df_EmbedInventories.shape}")
        dimred_time = time.time()
        if self.n_components is not None:
            if self.n_components >= self.df_EmbedInventories.shape[1]:
                whi(f"Dimension reduction not needed: {self.df_EmbedInventories.shape}")
                self.n_components = "OrigD"
                if self.dimred_method != "pca":
                    raise IgnoreInGrid(red("No dimension reduction needed and not asking to use PCA, this probably mean that the result would be redundant. Exiting now."))
            elif self.dimred_method == "pca":
                whi("Applying PCA")
                pca = PCA(
                        n_components=self.n_components,
                        copy=True,
                        svd_solver="full",
                        )
                self.df_EmbedInventories = pd.DataFrame(data=pca.fit_transform(self.df_EmbedInventories.values), index=index, dtype=np.float32)
                whi(f"EVR: {np.sum(pca.explained_variance_ratio_)}")
            elif self.dimred_method == "nmf":
                whi("Applying NMF")
                nmf = NMF(
                        n_components=self.n_components,
                        init=None,
                        random_state=42,
                        verbose=self.verbose,
                        )
                self.df_EmbedInventories = pd.DataFrame(data=nmf.fit_transform(self.df_EmbedInventories.values), index=index, dtype=np.float32)
                whi(f"Reconstruction error: {nmf.reconstruction_err_}")
            elif self.dimred_method == "umap":
                whi("Applying UMAP")
                n_n = int(len(self.df_EmbedInventories) * (100-15) / (1000 - 100))
                n_n = min(max(n_n, 15), 100)
                red(f"Will use n_neighbors={n_n}")
                umap_kwargs = {
                        "n_jobs": -1,
                        "verbose": self.verbose,
                        "metric": "cosine",
                        "n_components": self.n_components,
                        # the initial position is the nD PCA
                        # "init": PCA(
                        #     n_components=self.n_components,
                        #     random_state=42).fit_transform(self.df_EmbedInventories),
                        "init": NMF(
                            n_components=self.n_components,
                            random_state=42,
                            verbose=self.verbose,
                            ).fit_transform(self.df_EmbedInventories),
                        #"transform_seed": 42,
                        #"random_state": 42,  # turns off some multithreading section of the code
                        "n_neighbors":  n_n,
                        #"min_dist": 0.01,
                        "low_memory":  True,
                        #"densmap": True,  # try to preserve local density
                        "n_epochs": 1000,  # None will automatically adjust
                        "target_metric": self.norm,  # not sure what it does
                        }
                self.df_EmbedInventories = pd.DataFrame(
                        data=umap.UMAP(**umap_kwargs).fit_transform(self.df_EmbedInventories.values),
                        index=index,
                        dtype=np.float32,
                        )
            elif self.dimred_method == "dictionnarylearning":
                whi("Applying Dictionnary Learning")
                dico = DictionaryLearning(
                        n_components=self.n_components,
                        n_jobs=-1,
                        verbose=self.verbose,
                        random_state=42,
                        transform_algorithm="lasso_lars",
                        transform_alpha=0.1,
                        )
                data = dico.fit(self.df_EmbedInventories.values).transform(self.df_EmbedInventories.values)
                self.df_EmbedInventories = pd.DataFrame(data=data, index=index, dtype=np.float32)
            elif self.dimred_method == "bvae":
                whi("Applying Variational Autoencoder")
                def bvae_dim_reduc(n_components, dataset):
                    bvae = OptimizedBVAE(
                            input_dim=dataset.shape[1],
                            z_dim=n_components,
                            dataset_size=dataset.shape[0],
                            use_VeLO=False,
                            variational=False,
                            )
                    bvae = bvae.fit(dataset=dataset)
                    proj = bvae.transform(dataset)
                    return proj
                bvae_cache = Memory("cache/BVAE", verbose=1)
                cached_bvaedimreduc = bvae_cache.cache(bvae_dim_reduc)
                self.df_EmbedInventories = pd.DataFrame(
                        data=cached_bvaedimreduc(
                            n_components=self.n_components,
                            dataset=self.df_EmbedInventories.values
                            ),
                        index=index,
                        dtype=np.float32)
            else:
                raise ValueError(f"Invalid dimred_method: '{self.dimred_method}'")
        else:
            self.n_components = "OrigD"
            whi("Dimension reduction not needed")
            if self.dimred_method != "pca":
                raise IgnoreInGrid(red("No dimension reduction needed and not asking to use PCA, this probably mean that the result would be redundant. Exiting now."))
        whi(f"Current self.df_EmbedInventories shape: {self.df_EmbedInventories.shape}")

        dimred_time = time.time() - dimred_time

        # apply the clustering method for an increasing value of k to
        # construct the plot
        whi("Creating the intrinsic metrics plot")
        storage = pd.DataFrame(
                data=[],
                index=self.n_cluster,
                columns=[
                    "intrinsic_metrics_names",
                    "davies_bouldin_score",
                    "calinski_harabasz_score",
                    "silhouette_score",
                    "clustering_duration",
                    "n_cluster",
                    "cluster_method",
                    "mode",
                    "note",
                    "norm",
                    "dimred_method",
                    "n_components",
                    "df_shape",
                    "dimred_duration",
                    ])
        storage["intrinsic_metrics_names"] = json.dumps([
                "davies_bouldin_score",
                "calinski_harabasz_score",
                "silhouette_score",
                ])
        storage["n_cluster"] = self.n_cluster
        storage["cluster_method"] = self.cluster_method
        storage["mode"] = self.mode
        storage["note"] = self.note
        storage["norm"] = self.norm
        storage["dimred_method"] = self.dimred_method
        storage["n_components"] = self.n_components
        storage["dimred_duration"] = dimred_time
        storage["df_shape_nid"] = self.df_EmbedInventories.shape[0]
        storage["df_shape_ndim"] = self.df_EmbedInventories.shape[1]
        multi_index = pd.MultiIndex.from_product(
                [
                    self.df_EmbedInventories.index.tolist(),
                    self.n_cluster,
                     ], names=["subject", "n_cluster"])
        predictions = pd.DataFrame(
                index=multi_index,
                columns=["prediction"],
                data=np.nan,
                )

        precomputed_affinity = None

        for k in tqdm(storage.index, desc=f"Applying {self.cluster_method}", unit=" k"):
            if psutil.swap_memory().used != 0:
                tqdm.write("Fixing swap")
                os.system("sudo swapoff -av && sudo swapon -av")
                tqdm.write("Swap fixed.")

            if self.verbose:
                tqdm.write(f"Applying {self.cluster_method} for k={k}")
            start_time = time.time()
            if self.cluster_method == "kmeans":
                clst = KMeans(
                        n_clusters=k,
                        n_init=10,
                        verbose=self.verbose,
                        copy_x=True,
                        )
                pred = clst.fit_predict(self.df_EmbedInventories)
                storage.loc[k, "inertia"] = clst.inertia_

            elif self.cluster_method == "spectralcosine":
                if precomputed_affinity is not None:  # reuse affinity matrix
                    clst = SpectralClustering(
                            n_clusters=k,
                            n_jobs=-1,
                            assign_labels="kmeans",
                            n_init=10,
                            affinity="precomputed",
                            verbose=self.verbose,
                            )
                    whi("Reusing affinity matrix.")
                    pred = clst.fit_predict(precomputed_affinity)
                else:  # compute affinity matrix
                    clst = SpectralClustering(
                            n_clusters=k,
                            n_jobs=-1,
                            assign_labels="kmeans",
                            n_init=10,
                            affinity="rbf",
                            verbose=self.verbose,
                            )
                    whi("Computing affinity matrix.")
                    pred = clst.fit_predict(self.df_EmbedInventories)
                    precomputed_affinity = clst.affinity_matrix_
            elif self.cluster_method == "bisectingkmeans":
                clst = BisectingKMeans(
                        n_clusters=k,
                        init="k-means++",
                        n_init=10,
                        random_state=42,
                        verbose=self.verbose,
                        copy_x=True,
                        bisecting_strategy="biggest_inertia",
                        )
                pred = clst.fit_predict(self.df_EmbedInventories)
                storage.loc[k, "inertia"] = clst.inertia_
            else:
                raise ValueError("invalid cluster_method")

            n_pred_values = len(list(set(list(pred))))
            assert n_pred_values <= k, f"found {n_pred_values} different labels instead of <={k}"
            if n_pred_values < k:
                tqdm.write("Predictions: " + ",".join([str(x) for x in set(pred)]))
                red(f"Suspicious prediction labels? k={k} but found {n_pred_values} different labels")

            storage.loc[k, "clustering_duration"] = time.time() - start_time

            for subject, val in zip(index, pred):
                predictions.loc[ (subject, k), "prediction"] = val

            # whi("Computing davies bouldin score")
            db = davies_bouldin_score(
                    self.df_EmbedInventories,
                    pred)
            storage.loc[k, "davies_bouldin_score"] = db
            # whi("Computing calinski harabasz score")
            ch = calinski_harabasz_score(
                    self.df_EmbedInventories,
                    pred)
            storage.loc[k, "calinski_harabasz_score"] = ch
            # whi("Computing silhouette score")
            ss = silhouette_score(
                    self.df_EmbedInventories,
                    pred,
                    metric="euclidean")
            storage.loc[k, "silhouette_score"] = ss

            if self.verbose:
                tqdm.write(f"Done with K={k}")
            gc.collect()

        # get the correlation between each value and the inertia, to see if
        # it's going down or not
        for m in json.loads(storage.iloc[0]["intrinsic_metrics_names"]):
            # make sure the correlation is 0 to 1 instead of -1 to +1
            storage[f"COR_{m}"] = (storage[m].corr(storage["inertia"], method="kendall") + 1) / 2

        # store the results
        storage.to_pickle(
                str(self.resultdir / f"RES_{self.timestamp}.pickle"),
                )
        predictions.to_pickle(
                str(self.resultdir / f"PRED_{self.timestamp}.pickle"),
                )

        # save metric plot
        if not self.skip_plot:
            lines = []
            labels = []
            plt.figure(figsize=(16, 8))
            ax1 = plt.gca()
            ax1.set_xlabel('N Clusters')
            ax1.set_title(f'{self.mode} {self.cluster_method} {self.norm} {self.dimred_method} {self.n_components}'.title(), fontsize=20)

            # for inertia lower is better
            # if storage["inertia"].tolist():
            vals = ((storage["inertia"] - storage["inertia"].min()) / (storage["inertia"].max() - storage["inertia"].min())).tolist()
            line1, = ax1.plot(
                    self.n_cluster,
                    vals,
                    color="black",
                    marker="x")
            lines.append(line1)
            labels.append("Inertia")
            for i, v in enumerate(storage["inertia"]):
                ax1.annotate(
                        f"{v:.4f}",
                        xy=(self.n_cluster[i], vals[i]),
                        xytext=(-7, 7),
                        textcoords='offset points')

            # davies bouldin is 0 to +inf and lower is better
            vals = ((storage["davies_bouldin_score"] - storage["davies_bouldin_score"].min()) / (storage["davies_bouldin_score"].max() - storage["davies_bouldin_score"].min())).tolist()
            line2, = ax1.plot(
                    self.n_cluster,
                    vals,
                    label="DBS",
                    color="red",
                    marker="o")
            lines.append(line2)
            labels.append("Davies Bouldin")
            for i, v in enumerate(storage["davies_bouldin_score"]):
                ax1.annotate(
                        f"{v:.4f}",
                        xy=(self.n_cluster[i], vals[i]),
                        xytext=(-7, 7),
                        textcoords='offset points')

            # silhouette is -1 to +1 and higher is better
            vals = (1 - (storage["silhouette_score"] + 1 - (storage["silhouette_score"] + 1).min()) / ((storage["silhouette_score"] + 1).max() - (storage["silhouette_score"] + 1).min())).tolist()
            line3, = ax1.plot(
                    self.n_cluster,
                    vals,
                    label="SS",
                    color="blue",
                    marker="o")
            lines.append(line3)
            labels.append("Silhouette")
            for i, v in enumerate(storage["silhouette_score"]):
                ax1.annotate(
                        f"{v:.4f}",
                        xy=(self.n_cluster[i], vals[i]),
                        xytext=(-7, 7),
                        textcoords='offset points')

            # calinski harabasz is 0 to +inf and higher is better
            vals = (1 - ((storage["calinski_harabasz_score"] - storage["calinski_harabasz_score"].min()) / (storage["calinski_harabasz_score"].max() - storage["calinski_harabasz_score"].min()))).tolist()
            line4, = ax1.plot(
                    self.n_cluster,
                    vals,
                    label="CHS",
                    color="green",
                    marker="o")
            lines.append(line4)
            labels.append("Calinski Harabasz")
            for i, v in enumerate(storage["calinski_harabasz_score"]):
                ax1.annotate(
                        f"{v:.4f}",
                        xy=(self.n_cluster[i], vals[i]),
                        xytext=(-7, 7),
                        textcoords='offset points')

            ax1.legend(lines, labels)

            plt.tight_layout()
            plt.savefig(str(self.resultdir / f"KPLOT_{self.mode}_{self.cluster_method}_{self.norm}_{self.dimred_method}_{self.n_components}_{self.timestamp}.png"))
            if self.show_plot:
                whi("Showing plot.")
                plt.show()
            output_image = plt.gcf()
        else:
            output_image = None
        self.output = {
                "storage": storage,
                "predictions": predictions,
                "intrinsic_metrics_plot": output_image,
                "resultdir": self.resultdir,
                "sentence_embeddings": self.sentence_embeddings,
                "subjects_embeddings": self.df_EmbedInventories,
                "df_shape": self.df_EmbedInventories.shape,
                "self.df_text": str(self.df_text),
                "self.df_answ": str(self.df_answ),
                "n_cluster": self.n_cluster,
                }


if __name__ == "__main__":
    main = fire.Fire(QuestEA)
