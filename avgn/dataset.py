### create a dataset object given a folder full of JSON data

import numpy as np

from avgn.utils.paths import DATA_DIR, most_recent_subdirectory
from avgn.signalprocessing.filtering import prepare_mel_matrix
from avgn.utils.json import read_json
from avgn.utils.hparams import HParams
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed


class DataSet(object):
    """
    """

    def __init__(
        self, DATASET_ID, hparams=None, default_rate=None, build_mel_matrix=True
    ):
        self.default_rate = None

        if hparams is None:
            self.hparams = HParams()
        else:
            self.hparams = hparams

        self.DATASET_ID = DATASET_ID
        if type(self.DATASET_ID) == list:
            self.dataset_loc = [
                most_recent_subdirectory(DATA_DIR / "processed" / i) for i in DATASET_ID
            ]
        else:
            self.dataset_loc = most_recent_subdirectory(
                DATA_DIR / "processed" / DATASET_ID
            )
        self._get_wav_json_files()

        self.sample_json = read_json(self.json_files[0])

        self._load_datafiles()

        self._get_unique_individuals()

        if build_mel_matrix:
            self.build_mel_matrix()

    def _get_wav_json_files(self):
        """ find wav and json files in data folder
        """
        if type(self.dataset_loc) == list:
            self.wav_files = np.concatenate(
                [list((i / "WAV").glob("*.WAV")) for i in self.dataset_loc]
            )[: self.hparams.nex]
            self.json_files = np.concatenate(
                [list((i / "JSON").glob("*.JSON")) for i in self.dataset_loc]
            )[: self.hparams.nex]
        else:
            self.wav_files = list((self.dataset_loc / "WAV").glob("*.WAV"))[
                : self.hparams.nex
            ]
            self.json_files = list((self.dataset_loc / "JSON").glob("*.JSON"))[
                : self.hparams.nex
            ]

    def build_mel_matrix(self, rate=None):
        if rate is None:
            rate = self.sample_json["samplerate_hz"]
        self.mel_matrix = prepare_mel_matrix(self.hparams, rate)

    def _get_unique_individuals(self):
        self.json_indv = np.array(
            [
                value.data["indvs"].keys()
                for key, value in tqdm(
                    self.data_files.items(),
                    desc="getting unique individuals",
                    leave=False,
                )
            ]
        )
        self._unique_indvs = np.unique(self.json_indv)

    def _load_datafiles(self):
        with Parallel(
            n_jobs=self.hparams.n_jobs, verbose=self.hparams.verbosity
        ) as parallel:
            df = parallel(
                delayed(DataFile)(i) for i in tqdm(self.json_files, desc="loading json")
            )
            self.data_files = {i.stem: df for i, df in zip(self.json_files, df)}


class DataFile(object):
    """ An object corresponding to a json file
    """

    def __init__(self, json_loc):
        self.data = read_json(json_loc)
        self.indvs = list(self.data["indvs"].keys())

