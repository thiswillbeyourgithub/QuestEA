import os
import logging
from logging import handlers
from tqdm import tqdm
import openai
from pathlib import Path
import re
from transformers import AutoModel

import numpy as np
from sentence_transformers import SentenceTransformer


def coloured_log(color_asked, log):
    """used to print color coded logs"""
    col_red = "\033[91m"
    col_yel = "\033[93m"
    col_rst = "\033[0m"

    # all logs are considered "errors" otherwise the datascience libs just
    # overwhelm the logs

    if color_asked == "white":
        def printer(string, **args):
            if isinstance(string, list):
                string = ",".join(string)
            string = str(string)
            log.error(string)
            tqdm.write(col_rst + string + col_rst, **args)
            return string
    elif color_asked == "yellow":
        def printer(string, **args):
            if isinstance(string, list):
                string = ",".join(string)
            string = str(string)
            log.error(string)
            tqdm.write(col_yel + string + col_rst, **args)
            return string
    elif color_asked == "red":
        def printer(string, **args):
            if isinstance(string, list):
                string = ",".join(string)
            string = str(string)
            log.error(string)
            tqdm.write(col_red + string + col_rst, **args)
            return string
    return printer


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    To control logging level for various modules used in the application:
    https://github.com/huggingface/transformers/issues/3050#issuecomment-682167272
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.search(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def _get_sentence_encoder(mode, cache, normalizer):
    """Load either the sbert model or the openai API.
    Caching is used.
    """
    assert mode.startswith("llm_"), f"unexpected mode: '{mode}'"
    if mode == "llm_openai":
        assert Path("API_KEY.txt").exists(), "No api key found"
        openai.api_key = str(Path("API_KEY.txt").read_text()).strip()
        model_name = "text-embedding-ada-002"

        if cache is None:
            cached_encoder = openai.Embedding.create
        else:
            cached_encoder = cache.cache(openai.Embedding.create)
        def sentence_encoder(sentences):
            vectors = openai_sentence_encoder(
                    sentences,
                    vectorizer=cached_encoder,
                    model_name=model_name,
                    ).squeeze()
            if len(vectors.shape) == 1:
                return normalizer.fit_transform(vectors.reshape(1, -1))
            else:
                return normalizer.fit_transform(vectors)

    elif mode.startswith("llm_random"):
        # fake llm that returns normalized random vectors
        # syntax example mode = "llm_random_500" to create vector of 500 dimensions
        assert mode.startswith("llm_random_"), f"invalid random mode {mode}"
        n = mode.replace("llm_random_", "")
        assert n.isdigit(), f"can't extract dimension number from {mode}'"
        n = int(n)

        def sentence_encoder(sentences):
            np.random.seed(42)
            vectors = np.random.rand(len(sentences), n).squeeze()
            np.random.seed(None)
            if len(vectors.shape) == 1:  # needed for the test vector
                return normalizer.fit_transform(vectors.reshape(1, -1))
            else:
                return normalizer.fit_transform(vectors)

    elif "jina-embeddings" in mode:
        model_name = mode[4:]
        try:
            # trust_remote_code is needed to use the encode method
            model = AutoModel.from_pretrained(
                    'jinaai/' + model_name,
                    trust_remote_code=True)
        except Exception as err:
            red(f"Error when loading hugging face model '{model_name}': '{err}'")
            raise

        if cache is None:
            cached_encoder = model.encode
        else:
            cached_encoder = cache.cache(model.encode)

        def sentence_encoder(sentences):
            vectors = cached_encoder(
                    sentences=sentences,
                    max_length=2048,  # if not set the model will crash
                    ).squeeze()
            if len(vectors.shape) == 1:  # fix for the test vector
                return normalizer.fit_transform(vectors.reshape(1, -1))
            else:
                return normalizer.fit_transform(vectors)

    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        model_name = mode[4:]
        # load cache sentence embedding model
        try:
            model = SentenceTransformer(model_name)
        except Exception as err:
            red(f"Error when loading sbert model '{model_name}': '{err}'")
            raise

        if cache is None:
            cached_encoder = model.encode
        else:
            cached_encoder = cache.cache(model.encode)
        def sentence_encoder(sentences):
            vectors = sbert_sentence_encoder(
                    sentences,
                    model=model,
                    vectorizer=cached_encoder,
                    ).squeeze()
            if len(vectors.shape) == 1:  # fix for the test vector
                return normalizer.fit_transform(vectors.reshape(1, -1))
            else:
                return normalizer.fit_transform(vectors)

    return sentence_encoder


def sbert_sentence_encoder(sentences, model, vectorizer):
    """sbert silently crops any token above the max_seq_length,
    so we do a windowing embedding then maxpool.
    The normalization happens afterwards."""
    max_len = model.get_max_seq_length()

    if not isinstance(max_len, int):
        # the clip model has a different way to use the encoder
        # sources : https://github.com/UKPLab/sentence-transformers/issues/1269
        assert "clip" in str(model).lower(), f"sbert model with no 'max_seq_length' attribute and not clip: '{model}'"
        max_len = 77
        encode = model._first_module().processor.tokenizer.encode
    else:
        if hasattr(model.tokenizer, "encode"):
            # most models
            encode = model.tokenizer.encode
        else:
            # word embeddings models like glove
            encode = model.tokenizer.tokenize

    assert isinstance(max_len, int), "n must be int"
    n23 = (max_len * 2) // 3
    add_sent = []  # additional sentences
    add_sent_idx = []  # indices to keep track of sub sentences

    for i, s in enumerate(sentences):
        # skip if the sentence is short
        length = len(encode(s))
        if length <= max_len:
            continue

        # otherwise, split the sentence at regular interval
        # then do the embedding of each
        # and finally maxpool those sub embeddings together
        # the renormalization happens later in the code
        sub_sentences = []
        words = s.split(" ")
        avg_tkn = length / len(words)
        j = int(max_len / avg_tkn * 0.8)  # start at 90% of the supposed max_len
        while len(encode(" ".join(words))) > max_len:

            # if reached max length, use that minus one word
            until_j = len(encode(" ".join(words[:j])))
            if until_j >= max_len:
                jjj = 1
                while len(encode(" ".join(words[:j-jjj]))) >= max_len:
                    jjj += 1
                sub_sentences.append(" ".join(words[:j-jjj]))

                # remove first word until 1/3 of the max_token was removed
                # this way we have a rolling window
                jj = int((max_len // 3) / avg_tkn * 0.8)
                while len(encode(" ".join(words[jj:j-jjj]))) > n23:
                    jj += 1
                words = words[jj:]

                j = int(max_len / avg_tkn * 0.8)
            else:
                diff = abs(max_len - until_j)
                if diff > 10:
                    j += max(1, int(10 / avg_tkn))
                else:
                    j += 1

        sub_sentences.append(" ".join(words))

        sentences[i] = " "  # discard this sentence as we will keep only
        # the sub sentences maxpooled

        # remove empty text just in case
        if "" in sub_sentences:
            while "" in sub_sentences:
                sub_sentences.remove("")
        assert sum([len(encode(ss)) > max_len for ss in sub_sentences]) == 0, f"error when splitting long sentences: {sub_sentences}"
        add_sent.extend(sub_sentences)
        add_sent_idx.extend([i] * len(sub_sentences))

    if add_sent:
        sent_check = [
                len(encode(s)) > max_len
                for s in sentences
                ]
        addsent_check = [
                len(encode(s)) > max_len
                for s in add_sent
                ]
        assert sum(sent_check + addsent_check) == 0, (
            f"The rolling average failed apparently:\n{sent_check}\n{addsent_check}")
    vectors = vectorizer(
            sentences=sentences + add_sent,
            show_progress_bar=True,
            output_value="sentence_embedding",
            convert_to_numpy=True,
            normalize_embeddings=False,
            )

    if add_sent:
        # at the position of the original sentence (not split)
        # add the vectors of the corresponding sub_sentence
        # then return only the 'maxpooled' section
        assert len(add_sent) == len(add_sent_idx), (
            "Invalid add_sent length")
        offset = len(sentences)
        for sid in list(set(add_sent_idx)):
            id_range = [i for i, j in enumerate(add_sent_idx) if j == sid]
            add_sent_vec = vectors[
                    offset + min(id_range): offset + max(id_range), :]
            vectors[sid] = np.amax(add_sent_vec, axis=0)
        return vectors[:offset]
    else:
        return vectors


def openai_sentence_encoder(sentences, vectorizer, model_name):
    out = vectorizer(
            input=sentences,
            model=model_name,
            )
    return np.array([x["embedding"] for x in out['data']])


# adds logger file, restrict it to X lines
log_formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
file_handler = handlers.RotatingFileHandler(
        "logs.txt",
        mode='a',
        maxBytes=1000000,
        backupCount=3,
        encoding=None,
        delay=0,
        )
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(file_handler)


whi = coloured_log("white", log)
yel = coloured_log("yellow", log)
red = coloured_log("red", log)

class IgnoreInGrid(Exception):
    pass
