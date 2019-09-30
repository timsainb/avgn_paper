import numpy as np

from avgn.utils.paths import DATA_DIR, most_recent_subdirectory
from avgn.signalprocessing.filtering import prepare_mel_matrix
from avgn.utils.json import read_json
from avgn.utils.hparams import HParams


class DataSet(object):
    """
    """

    def __init__(self, DATASET_ID, hparams=None, default_rate=None, load_jsons=True):
        self.default_rate = None
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
        self._get_unique_individuals()
        self.sample_json = read_json(self.json_files[0])
        self.data_files = {
            i.stem: DataFile(i, load_jsons=load_jsons) for i in self.json_files
        }

        if hparams is None:
            self.hparams = HParams()
        else:
            self.hparams = hparams

        self.build_mel_matrix()

    def _get_wav_json_files(self):
        if type(self.dataset_loc) == list:
            self.wav_files = np.concatenate(
                [list((i / "WAV").glob("*.WAV")) for i in self.dataset_loc]
            )
            self.json_files = np.concatenate(
                [list((i / "JSON").glob("*.JSON")) for i in self.dataset_loc]
            )
        else:
            self.wav_files = list((self.dataset_loc / "WAV").glob("*.WAV"))
            self.json_files = list((self.dataset_loc / "JSON").glob("*.JSON"))

    def build_mel_matrix(self, rate=None):
        if rate is None:
            rate = self.sample_json["samplerate_hz"]
        self.mel_matrix = prepare_mel_matrix(self.hparams, rate)

    def _get_unique_individuals(self):
        self.json_indv = np.array(
            [list(read_json(i)["indvs"].keys()) for i in self.json_files]
        )
        self._unique_indvs = np.unique(self.json_indv)


class DataFile(object):
    """ An object corresponding to a json file
    """

    def __init__(self, json_loc, load_jsons=True):
        self.data = read_json(json_loc)
        self.indvs = list(self.data["indvs"].keys())

