import scipy.io
import re
import os
import ftfy
from bs4 import BeautifulSoup
from pathlib import Path
from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy as np

# different import if imported or not
try:
    from .misc import whi, red
except Exception as err:
    print(f"Exception when loading from .misc: '{err}'")
    from misc import whi, red

fix = ftfy.fix_text

dataset_list = [
        "AdolescentDepressionSRQA",
        "AdolescentDepressionCU",
        "RutledgeSmartphone",
        "16PF",
        "DASS",
        "Hamilton",
        "HEXACO",
        "IPIP",
        ]


class Dataloader:
    def __init__(
            self,
            datasetname,
            mode,
            norm_f,
            ):
        """
        takes as input the datasetname, the mode and the norm.

        Creates a df_text pandas dataframe that contains as values the sentences
            that must be embedded. The id can be anything but has to match
            the columns of df_answ.
        Then creates df_answ that contains the actual inventory answer. The
            column names have to be the same as the id of df_text.

        A L1 or L2 norm will be applied to the value depending on the argument.

        The mode refers to the some post processing that can be requested.
        For example for the HEXACO dataset you can use "feat_raw" which
            means that each columns will contain the normalized raw values of
            the inventories. Or "feat_agg" which means that the
            "HEXACO dimensions" will be computed, normalized and be used
            as dimensions.
        """
        if isinstance(norm_f, str):
            norm_f = Normalizer(norm_f)
        if "_no_norm" in mode:
            assert mode.startswith("feat_"), f"unexpected mode {mode}"
            self.mode = mode.replace("_no_norm", "")
            self.skip_norm = True
        else:
            self.mode = mode
            self.skip_norm = False
        if datasetname not in dataset_list:
            raise Exception(f"{datasetname} not in {dataset_list}")

        self.datasetdir = Path("datasets")
        assert self.datasetdir.exists(), "No datasets folder found"
        datadir_ = [str(f) for f in self.datasetdir.iterdir()]
        assert len([f for f in datadir_ if datasetname in f]) != 0, (
                "dataset folder not found")
        assert len([f for f in datadir_ if datasetname in f]) == 1, (
                "more than 1 dataset folder found")

        self.targetdatapath = Path([f for f in datadir_ if datasetname in f][0])
        whi(f"Loading dataset {datasetname}")

        self.norm_f = norm_f

        for name in dataset_list:
            if name in str(self.targetdatapath):
                self.df_text = self._load_inventory_text(name=name)
                self.df_answ = self._load_and_process_inventory_answers(name=name)
                break

        # check conformity
        assert hasattr(self, "df_text"), "df_text failed to load"
        assert not self.df_answ.empty, "empty df_answ"
        if self.mode == "feat_raw":
            assert sorted(self.df_answ.columns.tolist()) == sorted(
                    self.df_text.index.tolist()), (
                            "columns of df_answ don't match df_text index")

        if self.mode in ["feat_agg", "feat_raw"]:
            if len(self.df_answ.values.squeeze().shape) == 1:
                # allow 1d df by projecting to unit circle otherwise
                # some clustering technique will fail
                # example: rutledge depression score
                self.df_answ /= self.df_answ.max() / 2
                self.df_answ -= 1
                angles = np.arccos(self.df_answ)
                self.df_answ = pd.DataFrame(
                        data=np.array([np.cos(angles), np.sin(angles)]).squeeze().T,
                        index=self.df_answ.index.tolist(),
                        columns=["x", "y"],
                        dtype=np.float32)
            if not self.skip_norm:
                self.df_answ[:] = self.norm_f.fit_transform(self.df_answ.values)
        else:  # make sure that the minimum value is not 0 for the LLM
            assert self.mode.startswith("llm_"), "unexpected mode"
            assert not self.df_text.empty, "empty df_text"
            assert np.isclose((self.df_answ.values.ravel() % 1).sum(), 0), "answers to inventory is supposed to be integers"

            if self.df_answ.min().min() == 0:
                self.df_answ += 1
            assert self.df_answ.min().min() >= 1

        assert not self.df_answ.isna().values.ravel().any(), "df_answ contains na"
        assert not self.df_text.isna().values.ravel().any(), "df_text contains na"
        assert not self.df_answ.sum(axis=1).min() == 0, "df_answ contains rows that sum to 0"

        if not self.df_answ.values.dtype == "float32":
            self.df_answ = self.df_answ.astype("float32")

        return

    def _load_inventory_text(
            self,
            name,
            ):
        whi(f"Loading code for {name}")
        assert self.targetdatapath.exists(), f"{self.targetdatapath} not found"

        if name == "IPIP":
            code_file = self.targetdatapath / "codebook.txt"
            lines = fix(code_file.read_text()).split("\n")
            codes = lines[7:57]
            assert len(codes) == 50, "invalid code length"
            data = [x.split("\t") for x in codes]
            data = [[i, enhance_phrasing(s)] for i, s in data]
            df_text = pd.DataFrame(
                    data=data,
                    columns=["id", "sentence"],
                    ).set_index("id")

        elif name == "DASS":
            code_file = self.targetdatapath / "codebook.txt"
            lines = fix(code_file.read_text()).split("\n")
            codes = lines[8:50]  # + lines[72:82]  72 to 82 is TIPI
            assert len(codes) == 42, "invalid code length"
            data = [x.split("\t") for x in codes]
            data = [[i.strip(), enhance_phrasing(s)] for i, s in data]
            for i in range(len(data)):
                assert len(data[i]) == 2, "invalid data length"
                #if data[i][0].startswith("TIPI"):
                #    data[i][1] = "I see myself as: '" + data[i][1] + "'"
            df_text = pd.DataFrame(
                    data=data,
                    columns=["id", "sentence"],
                    ).set_index("id")

        elif name == "HEXACO":
            code_file = self.targetdatapath / "codebook.txt"
            lines = fix(code_file.read_text()).split("\n")
            codes = lines[3:243]
            assert len(codes) == 240, "invalid code length"
            data=[[fix(x.split(" ")[0]).strip(), enhance_phrasing(" ".join(x.split(" ")[1:]))] for x in codes]
            df_text = pd.DataFrame(
                    data=data,
                    columns=["id", "sentence"],
                    ).set_index("id")

        elif name == "16PF":
            code_file = self.targetdatapath / "codebook.html"
            html = code_file.read_text()
            soup = BeautifulSoup(html, 'html.parser')
            lines = soup.find_all("tr")
            codes = lines[1:163]
            data = [
                    [
                        fix(c.find_all("td")[0].text),
                        enhance_phrasing(c.find_all("td")[2].text.split("\"")[1]),
                        ]
                    for c in codes
                    ]
            df_text = pd.DataFrame(
                    data=data,
                    columns=["id", "sentence"],
                    ).set_index("id")
            assert len(codes) == 162, "invalid code length"

        elif name == "Hamilton":
            codebookdocx = self.targetdatapath / "Screening_Hamilton_Codebook.docx"
            codebookmd = self.targetdatapath / "Screening_Hamilton_Codebook.md"
            if (self.targetdatapath / "success").exists():
                (self.targetdatapath / "success").unlink()
            os.system(f"pandoc '{codebookdocx}' -o '{codebookmd}' && touch {self.targetdatapath}/success")
            assert (self.targetdatapath / "success").exists(), (
                "pandoc command failed?")
            lines = codebookmd.read_text().split("\n")[21:]

            serials = [""]
            question = [""]
            values = [""]
            for l in lines:
                l = l.strip()
                if l == "":
                    continue

                if "-----" in l:
                    serials.append("")
                    question.append("")
                    values.append("")
                    continue

                assert "|" in l, "| not found"
                sp = l.strip().split("|")
                while "" in sp:
                    sp.remove("")
                assert len(sp) == 3, "invalid length"
                serials[-1] += sp[0]
                question[-1] += sp[1]
                values[-1] += sp[2]

            serials = [s.strip() for s in serials[:-1]]
            question = [s.strip() for s in question[:-1]]
            values = [s.strip() for s in values[:-1]]

            while "" in serials:
                n = serials.index("")
                question.pop(n)
                values.pop(n)
                serials.pop(n)

            question = [enhance_phrasing(s) for s in question]

            assert len(serials) == len(values), "unequal length"
            assert len(values) == len(question), "unequal length"

            df_text = pd.DataFrame(
                    columns=["id", "sentence"],
                    data=[[k, v] for k,v in zip(serials, question)],
                    ).set_index("id")
            # buggy column:
            df_text = df_text.drop(index=["10010"])

        elif name == "AdolescentDepressionSRQA":
            content = (self.targetdatapath / "codes_srqa.txt").read_text().split("\n")
            for i, line in enumerate(content):
                if not line:
                    content[i] = None
                    continue
                if not line[0].isdigit():
                    ii = 1
                    while content[i-ii] is None:
                        ii += 1
                    content[i-ii] += line
                    content[i] = None
            content = [c for c in content if c is not None]
            assert len(content) == 20, "there are normally 20 sentences"
            sentences = []
            for i, c in enumerate(content):
                assert re.match("\d+\.\s?\w+", c), f"invalid code: '{c}'"
                while c[0].isdigit() or c[0] == ".":
                    c = c[1:].strip()
                while "  " in c:
                    c = c.replace("  ", " ")
                sentences.append(enhance_phrasing(c))

            # SRQ-A = social reward questionnaire adolescent version
            # https://www.researchgate.net/publication/348753985_Social_Reward_Questionnaire_-_Adolescent_version_SRQ-A
            indexes = ['SRQA1', 'SRQA2', 'SRQA3', 'SRQA4',
                       'SRQA5', 'SRQA6', 'SRQA7', 'SRQA8', 'SRQA9', 'SRQA10', 'SRQA11',
                       'SRQA12', 'SRQA13', 'SRQA14', 'SRQA15', 'SRQA16', 'SRQA17', 'SRQA18',
                       'SRQA19', 'SRQA20'
                       ]

            df_text = pd.DataFrame(
                    data=sentences,
                    index=indexes,
                    columns=["sentence"],
                    )
            df_text.index.name = "id"

        elif name == "AdolescentDepressionCU":
            content = (self.targetdatapath / "codes_icu.txt").read_text().split("\n")[1:]
            assert content[0].startswith("3. ")
            sentences = []
            for i, line in enumerate(content):
                if not line.strip():
                    continue
                while line[0].isdigit():
                    line = line[1:]
                while line[0] == ".":
                    line = line[1:]
                assert line.startswith(" ")
                line = line.strip()
                sentences.append(line)
            assert len(sentences) == 6, "there are normally 6 sentences"
            sentences = [enhance_phrasing(s) for s in sentences]

            # CU = callous unemotional traits
            # https://arc.psych.wisc.edu/self-report/the-inventory-of-callous-unemotional-traits-icu/
            indexes = [f"CU{i}" for i in range(1,7)]

            df_text = pd.DataFrame(
                    data=sentences,
                    index=indexes,
                    columns=["sentence"],
                    )
            df_text.index.name = "id"

        elif name == "RutledgeSmartphone":
            assert (self.targetdatapath / "BDIa_2015.txt").exists(), "missing code"
            content = (self.targetdatapath / "BDIa_2015.txt").read_text()

            temp = content.replace("1\n2\n3\n", "")
            sentences = {}
            for i in range(21, 0, -1):
                spl = temp.split(str(i))
                new = spl[-1]
                new = re.sub(r"\d", "", new).strip()
                if i == 21:
                    new = new.replace("I am purposely trying to lose weight.\nYes _____ No _____", "")
                sentences[i] = enhance_phrasing(new.strip())
                temp = "".join(spl[:-1])
            df_text = pd.DataFrame(
                    data=sentences.values(),
                    index=[i-1 for i in list(sentences.keys())],
                    columns=["sentence"],
                    )
            df_text.index.name = "id"

        else:
            raise ValueError(f"Invalid df_text name: {name}")

        whi("Done loading df_text")

        return df_text

    def _load_and_process_inventory_answers(
            self,
            name,
            ):
        if name == "IPIP":
            df_answ = self._load_dataset_big5()
        elif name == "DASS":
            df_answ = self._load_dataset_DASS()
        elif name == "HEXACO":
            df_answ = self._load_dataset_HEXACO()
        elif name == "Hamilton":
            df_answ = self._load_dataset_Hamilton()
        elif name == "16PF":
            df_answ = self._load_dataset_16PF()
        elif name == "AdolescentDepressionSRQA":
            df_answ = self._load_dataset_AdolescentDepressionSRQA(invent="SRQA")
        elif name == "AdolescentDepressionCU":
            df_answ = self._load_dataset_AdolescentDepressionSRQA(invent="CU")
        elif name == "RutledgeSmartphone":
            df_answ = self._load_dataset_RutledgeSmartphone()
        else:
            raise ValueError(f"Invalid preprocess data name: {name}")
        whi("Done loading df_answ\n")
        return df_answ

    def _load_dataset_RutledgeSmartphone(self):
        assert (self.targetdatapath / "Rutledge_GBE_risk_data_TOD.mat").exists(), "missing dataset"
        mat = scipy.io.loadmat(str(self.targetdatapath / "Rutledge_GBE_risk_data_TOD.mat"))
        dep = pd.DataFrame(mat["depData"].reshape(-1))
        dep = dep.dropna()
        dep = dep["bdiRaw"].apply(lambda x: x[0])
        dep = dep.dropna()

        # TODO: filter the subjects to keep highest quality

        df_answ = pd.DataFrame(
                data=dep.tolist(),
                dtype=np.float32,
                )
        df_answ = df_answ.dropna()

        # remove those with too few answers
        mu = df_answ.sum(axis=1).mean()
        sig = df_answ.sum(axis=1).std()
        df_answ = df_answ.drop(index=df_answ[df_answ.sum(axis=1) < mu - (sig / 2)].index)
        # and too many yes checked
        df_answ = df_answ.drop(index=df_answ[df_answ.sum(axis=1) > mu + 2 * sig].index)

        if self.mode == "feat_agg":
            df_answ = pd.DataFrame(
                    data=df_answ.sum(axis=1).values.squeeze(),
                    index=df_answ.index.tolist(),
                    dtype=np.float32)
        return df_answ

    def _load_dataset_AdolescentDepressionSRQA(self, invent):
        assert (self.targetdatapath / "Adolescent_SRQ_data.sav").exists(), (
            "sav file not found")
        df_answ = pd.read_spss(
                self.targetdatapath / "Adolescent_SRQ_data.sav",
                convert_categoricals=False,
                )

        df_answ = df_answ.dropna()
        if self.mode == "feat_raw" or self.mode.startswith("llm_"):
            df_answ = df_answ[self.df_text.index.tolist()]
        elif self.mode == "feat_agg":
            if invent == "SRQA":
                agg_index = ['SRQ_admir', 'SRQ_negsoc', 'SRQ_pass', 'SRQ_prosoc', 'SRQ_sociab']
            elif invent == "CU":
                agg_index = ['CU_total']
            self.df_text = self.df_text.drop(columns=[
                c for c in self.df_text.columns
                if c not in agg_index
                ])
            df_answ = pd.DataFrame(
                    index=df_answ.index,
                    columns=agg_index,
                    data=df_answ.loc[:, agg_index].values,
                    dtype=np.float32,
                    )
        return df_answ

    def _load_dataset_Hamilton(self):
        whi("Loading data file")

        assert (self.targetdatapath / "Screening_Hamilton_Final.sav").exists(), (
            "sav file not found")

        df_answ = pd.read_spss(
                self.targetdatapath / "Screening_Hamilton_Final.sav",
                convert_categoricals=False,
                )
        df_answ = df_answ.dropna()
        df_answ = df_answ.drop(columns=["PTNUM", "filter_$"])

        df_answ.columns = [x.replace("MA", "") for x in df_answ.columns.tolist()]
        df_answ.columns = [x.replace("S", "") for x in df_answ.columns.tolist()]

        # clean data ###############################
        whi("\n\nCleaning up data")
        whi(f"shape: {df_answ.shape}")

        df_answ = df_answ.drop(columns=["10010"])
        whi("Dropping column 10010 that is not parsed")
        df_answ = pd.DataFrame(df_answ, dtype=float)

        # remove those with too few answers
        mu = df_answ.sum(axis=1).mean()
        sig = df_answ.sum(axis=1).std()
        df_answ = df_answ.drop(index=df_answ[df_answ.sum(axis=1) < mu - (sig / 2)].index)
        # and too many yes checked
        df_answ = df_answ.drop(index=df_answ[df_answ.sum(axis=1) > mu + 2 * sig].index)

        if self.mode == "feat_raw" or self.mode.startswith("llm_"):
            # use the normalized raw answers as input for the kmeans
            df_answ = df_answ.drop(columns=["16000", "16001"])
            self.df_text = self.df_text.drop(index=["16000", "16001"])
            df_answ = pd.DataFrame(
                    index=df_answ.index,
                    columns=self.df_text.index,
                    data=df_answ.loc[:, self.df_text.index].values,
                    dtype=np.float32,
                    )
        elif self.mode == "feat_agg":
            df_answ = df_answ[["16000", "16001"]]
            self.df_text = self.df_text.loc[["16000", "16001"]]
            df_answ = pd.DataFrame(
                    index=df_answ.index,
                    columns=self.df_text.index,
                    data=df_answ.loc[:, self.df_text.index].values,
                    dtype=np.float32,
                    )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return df_answ

    def _load_dataset_16PF(self):
        df_text = self.df_text

        whi("Loading data file")

        assert (self.targetdatapath / "data.csv").exists(), (
            "data.csv not found")

        df_answ = pd.read_csv(self.targetdatapath / "data.csv", sep="\t")
        df_answ = df_answ.drop(columns=["country", "gender", "source"])
        cols = df_answ.columns.tolist()

        # clean data ###############################
        whi("\n\nCleaning up data")
        whi(f"shape: {df_answ.shape}")

        # remove na
        df_answ = df_answ.dropna()
        whi(f"no na: {df_answ.shape}")

        # keep only those reasonnable age
        df_answ = df_answ[df_answ.loc[:, 'age'] >= 18]
        df_answ = df_answ[df_answ.loc[:, 'age'] <= 80]
        whi(f"only good age: {df_answ.shape}")

        # remove if accuracy is low
        df_answ = df_answ[df_answ.loc[:, "accuracy"] >= 85]
        df_answ = df_answ[df_answ.loc[:, "accuracy"] <= 100]
        whi(f"not good accuracy: {df_answ.shape}")
        df_answ = df_answ.drop(columns=["accuracy", "age"])

        # keep only reasonnable speed of answers
        # columns with time spent answering questions
        # Calculate the total time for each survey
        df_answ["elapsed"] = pd.to_numeric(df_answ["elapsed"], errors="coerce")
        df_answ = df_answ.dropna()
        df_answ = df_answ[df_answ['elapsed'].between(100, 100000)]
        whi(f"reasonnable speed: {df_answ.shape}")
        df_answ = df_answ.drop(columns=["elapsed"])

        # remove invalid result
        df_answ = df_answ[(df_answ >= 1).all(axis=1)]
        df_answ = df_answ[(df_answ <= 5).all(axis=1)]
        whi(f"only correct answers: {df_answ.shape}")

        df_answ = df_answ.dropna()

        if self.mode == "feat_raw" or self.mode.startswith("llm_"):
            # use the normalized raw answers as input for the kmeans
            df_answ = pd.DataFrame(
                    index=df_answ.index,
                    columns=df_text.index,
                    data=df_answ.loc[:, df_text.index].values,
                    dtype=np.float32,
                    )
        elif self.mode == "feat_agg":
            # source: https://en.wikipedia.org/wiki/16PF_Questionnaire
            for primary_factor in ["A", "B", "C", "E", "F", "G", "H", "I",
                    "L", "M", "N", "O"]:
                cols = [c for c in df_answ.columns.tolist() if c.startswith(primary_factor)]
                df_answ.loc[:, "_" + primary_factor] = df_answ.loc[:, cols].sum(axis=1)

            to_drop = [c for c in df_answ.columns.tolist() if not c.startswith("_")]
            df_answ = df_answ.drop(columns=to_drop)
            df_answ = df_answ.rename(columns=lambda x: x.replace("_", ""))

            # optionnal, aggregate the primary factors into global factors
            factor_dict = {
                    "introextroversion": ["A", "F", "H", "N"],
                    "anxiety": ["C", "L", "O"],
                    "receptivity": ["A", "I", "M"],
                    "accomodation": ["E", "H", "L"],
                    "control": ["F", "G", "M"],
                    }
            for glob, primaries in factor_dict.items():
                cols = [c for c in df_answ.columns.tolist() if c in primaries]
                df_answ.loc[:, "_" + glob] = df_answ.loc[:, cols].sum(axis=1)
            to_drop = [c for c in df_answ.columns.tolist() if not c.startswith("_")]
            df_answ = df_answ.drop(columns=to_drop)
            df_answ = df_answ.rename(columns=lambda x: x.replace("_", ""))
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return df_answ

    def _load_dataset_HEXACO(self):
        df_text = self.df_text

        whi("Loading data file")

        assert (self.targetdatapath / "data.csv").exists(), (
            "data.csv not found")

        df_answ = pd.read_csv(self.targetdatapath / "data.csv", sep="\t")
        df_answ = df_answ.dropna()
        cols = df_answ.columns.tolist()

        HON = [col for col in cols if col.startswith("H")]
        EMO = [col for col in cols if col.startswith("E")]
        EXT = [col for col in cols if col.startswith("X")]
        AGR = [col for col in cols if col.startswith("A")]
        CON = [col for col in cols if col.startswith("C")]
        OPN = [col for col in cols if col.startswith("O")]
        dimensions = [HON, EMO, EXT, AGR, CON, OPN]
        question_col = []
        [question_col.extend(d) for d in dimensions]
        dimension_averages = ["honesty", "emotionality",
                              "extraversion", "agreeableness",
                              "conscientiousness", "openness"]
        whi("Computing aggregate score")
        for d in range(len(dimensions)):
            df_answ[dimension_averages[d]] = df_answ[dimensions[d]].mean(axis=1)

        # clean data ###############################
        whi("\n\nCleaning up data")
        whi(f"shape: {df_answ.shape}")

        # remove if V1 is not checked
        df_answ = df_answ[df_answ.loc[:, "V1"] == 7]
        df_answ = df_answ[df_answ.loc[:, "V2"] == 7]
        whi(f"not checked first question: {df_answ.shape}")
        df_answ = df_answ.drop(columns=["V1", "V2", "country"])

        # remove invalid result
        df_answ = df_answ[(df_answ.loc[:, question_col] >= 1).all(axis=1)]
        df_answ = df_answ[(df_answ.loc[:, question_col] <= 7).all(axis=1)]
        whi(f"only correct answers: {df_answ.shape}")

        # keep only reasonnable speed of answers
        # columns with time spent answering questions
        # Calculate the total time for each survey
        df_answ["elapse"] = pd.to_numeric(df_answ["elapse"], errors="coerce")
        df_answ = df_answ.dropna()
        df_answ = df_answ[df_answ['elapse'].between(100, 100000)]
        whi(f"reasonnable speed: {df_answ.shape}")
        df_answ = df_answ.drop(columns=["elapse"])

        if self.mode == "feat_raw" or self.mode.startswith("llm_"):
            # use the normalized raw answers as input for the kmeans
            df_answ = pd.DataFrame(
                    index=df_answ.index,
                    columns=df_text.index,
                    data=df_answ.loc[:, df_text.index].values,
                    dtype=np.float32,
                    )
        elif self.mode == "feat_agg":
            # use the normalized dimension average as input for the kmeans
            df_answ = pd.DataFrame(
                    index=df_answ.index,
                    columns=dimension_averages,
                    data=df_answ.loc[:, dimension_averages].values,
                    dtype=np.float32,
                    )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return df_answ

    def _load_dataset_DASS(self):
        df_text = self.df_text

        whi("Loading data file")

        assert (self.targetdatapath / "data.csv").exists(), (
            "data.csv not found")
        df_answ = pd.read_csv(self.targetdatapath / "data.csv", sep="\t")
        df_answ = df_answ.dropna()
        cols = df_answ.columns.tolist()
        question_col = [q for q in cols if q.startswith("Q") and q.endswith("A")]
        tipi_col = [q for q in cols if q.startswith("TIPI")]
        elapsed_col = [q for q in cols if (q.startswith("Q") or q.startswith("TIPI")) and q.endswith("E")]
        order_col = [q for q in cols if q.startswith("Q") and q.endswith("I")]
        to_drop = [q for q in cols if q not in question_col + elapsed_col] + order_col + elapsed_col + ["total_time"] + tipi_col

        # clean data ###############################
        whi("\n\nCleaning up data")
        whi(f"shape: {df_answ.shape}")

        # keep only those with 1 attempt
        df_answ = df_answ[df_answ.loc[:, 'uniquenetworklocation'] == 1]
        whi(f"only 1 trial: {df_answ.shape}")

        # keep only those reasonnable age
        df_answ = df_answ[df_answ.loc[:, 'age'] >= 18]
        df_answ = df_answ[df_answ.loc[:, 'age'] <= 80]
        whi(f"only good age: {df_answ.shape}")

        # remove invalid result
        df_answ = df_answ.loc[(df_answ.loc[:, question_col] >= 1).all(axis=1)]
        df_answ = df_answ.loc[(df_answ.loc[:, question_col] <= 4).all(axis=1)]
        df_answ = df_answ.loc[(df_answ.loc[:, tipi_col] >= 1).all(axis=1)]
        df_answ = df_answ.loc[(df_answ.loc[:, tipi_col] <= 7).all(axis=1)]
        whi(f"only correct answers: {df_answ.shape}")

        # only keep english speakers
        df_answ = df_answ.loc[df_answ.loc[:, "engnat"]==1]
        whi(f"only english speakers: {df_answ.shape}")

        # keep only reasonnable speed of answers
        # columns with time spent answering questions
        # Calculate the total time for each survey
        df_answ['total_time'] = df_answ[elapsed_col].sum(axis=1)
        df_answ = df_answ[df_answ['total_time'].between(10000, 1000000)]
        whi(f"reasonnable speed: {df_answ.shape}")

        df_answ = df_answ.drop(columns=to_drop)
        # remove the final A in Q35A etc
        df_answ.columns = [q[:-1] if q.endswith("A") else q for q in df_answ.columns]

        if self.mode == "feat_raw" or self.mode.startswith("llm_"):
            # use the normalized raw answers as input for the kmeans
            df_answ = pd.DataFrame(
                    index=df_answ.index,
                    columns=df_text.index,
                    data=df_answ.loc[:, df_text.index].values,
                    dtype=np.float32,
                    )
        elif self.mode == "feat_agg":
            subscale_dict = {
                    "Depression": [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42],
                    "Anxiety": [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41],
                    "Stress": [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]
                    }
            for scale, quest in subscale_dict.items():
                quest = [str(c) for c in quest]
                cols = [c for c in df_answ.columns
                        if (c not in subscale_dict.keys())
                        and str(c)[1:] in quest]
                df_answ.loc[:, scale] = df_answ.loc[:, cols].sum(axis=1)

            to_drop = [c for c in df_answ.columns if c.startswith("Q")]
            df_answ = df_answ.drop(columns=to_drop)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return df_answ

    def _load_dataset_big5(self):
        # positive questions adding to the trait.
        pos_questions = [
            'EXT1', 'EXT3', 'EXT5', 'EXT7', 'EXT9',                        # 5
            'EST1', 'EST3', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10',  # 8
            'AGR2', 'AGR4', 'AGR6', 'AGR8', 'AGR9', 'AGR10',                # 6
            'CSN1', 'CSN3', 'CSN5', 'CSN7', 'CSN9', 'CSN10',                # 6
            'OPN1', 'OPN3', 'OPN5', 'OPN7', 'OPN8', 'OPN9', 'OPN10',         # 7
        ]
        # negative (negating) questions subtracting from the trait.
        neg_questions = [
            'EXT2', 'EXT4', 'EXT6', 'EXT8', 'EXT10',  # 5
            'EST2', 'EST4',                        # 2
            'AGR1', 'AGR3', 'AGR5', 'AGR7',          # 4
            'CSN2', 'CSN4', 'CSN6', 'CSN8',          # 4
            'OPN2', 'OPN4', 'OPN6',                 # 3
        ]
        EXT = [col for col in pos_questions + neg_questions if col.startswith("EXT")]
        EST = [col for col in pos_questions + neg_questions if col.startswith("EST")]
        AGR = [col for col in pos_questions + neg_questions if col.startswith("AGR")]
        CSN = [col for col in pos_questions + neg_questions if col.startswith("CSN")]
        OPN = [col for col in pos_questions + neg_questions if col.startswith("OPN")]
        dimensions = [EXT, EST, AGR, CSN, OPN]
        dimension_averages = ["extraversion", "emotional_stability",
                              "agreeableness", "conscientiousness", "openness"]
        df_text = self.df_text

        whi("Loading data file")

        assert (self.targetdatapath / "data-final.csv").exists(), (
            "data-final.csv not found")
        df_answ = pd.read_csv(self.targetdatapath / "data-final.csv", sep="\t")
        df_answ = df_answ.dropna()

        # clean data ###############################
        whi("\n\nCleaning up data")
        whi(f"shape: {df_answ.shape}")

        # keep only those with 1 attempt
        df_answ = df_answ.loc[df_answ['IPC'] == 1]
        whi(f"only 1 trial: {df_answ.shape}")

        # remove invalid result
        df_answ = df_answ.loc[(df_answ[df_answ.columns.tolist()[:49]] >= 1).all(axis=1)]
        df_answ = df_answ.loc[(df_answ[df_answ.columns.tolist()[:49]] <= 5).all(axis=1)]
        whi(f"only correct answers: {df_answ.shape}")

        # keep only reasonnable speed of answers
        # columns with time spent answering questions
        qtime_cols = list(df_answ.columns)[50:100]
        # Calculate the total time for each survey
        df_answ['total_time'] = df_answ[qtime_cols].sum(axis=1)
        df_answ = df_answ[df_answ['total_time'].between(10000, 1000000)]
        whi(f"reasonnable speed: {df_answ.shape}")

        # Drop the redundant locations
        drop_cols = list(df_answ.columns[50:107])+['lat_appx_lots_of_err', 'long_appx_lots_of_err']
        df_answ = df_answ.drop((drop_cols), axis=1)
        whi(f"no duplicate location: {df_answ.shape}")

        # change answer formatting
        whi("Change answer format")
        # Replace the question answer with -2 to 2 scale depending if the question is positive or negative.
        # df_answ[pos_questions] = df_answ[pos_questions].replace({1:-2, 2:-1, 3:0, 4:1, 5:2})
        # df_answ[neg_questions] = df_answ[neg_questions].replace({1:2, 2:1, 3:0, 4:-1, 5:-2})

        # parse the negative questions differently
        #df_answ.loc[:, neg_questions] = 6 - df_answ.loc[:, neg_questions]

        # compute aggregate score
        whi("Computing aggregate score")
        for d in range(len(dimensions)):
            df_answ[dimension_averages[d]] = df_answ[dimensions[d]].mean(axis=1)

        if self.mode in "feat_raw" or self.mode.startswith("llm_"):
            # use the normalized raw answers as input for the kmeans
            df_answ = pd.DataFrame(
                    index=df_answ.index,
                    columns=df_text.index,
                    data=df_answ.loc[:, df_text.index].values,
                    dtype=np.float32,
                    )
        elif self.mode == "feat_agg":
            # use the normalized dimension average as input for the kmeans
            df_answ = pd.DataFrame(
                    index=df_answ.index,
                    columns=dimension_averages,
                    data=df_answ.loc[:, dimension_averages].values,
                    dtype=np.float32,
                    )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return df_answ

def enhance_phrasing(sentence):
    sentence = fix(sentence).strip()
    while "  " in sentence:
        sentence = sentence.replace("  ", " ")
    # sentence = f"One sentence to summarize myself could be '{sentence}'"
    if len(sentence.splitlines()) > 1:
        red(f"Sentence contained multiple lines: '{sentence}'")
    return sentence

if __name__ == "__main__":
    whi("Running tests for dataloader.")
    pd.set_option('display.max_colwidth', 8000)
    for norm_f in ["l1", "l2"]:
        for ds in dataset_list:
            for mode in ["feat_agg", "feat_raw", "llm_all-MiniLM-L6-v2"]:
                dl = Dataloader(
                        datasetname=ds,
                        norm_f=norm_f,
                        mode=mode,
                        )
                df_answ, df_text = dl.df_answ, dl.df_text
                print(f"{norm_f} {ds} {mode}")
                print("Questions:")
                print(df_text)
                print("Answers:")
                print(df_answ)
                print("=====================================")
