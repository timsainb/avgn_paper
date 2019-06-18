import pandas as pd
import re
from scipy.io import loadmat


def parse_marmoset_data(wavs, _filetype="wav"):
    """Parse filename of marmoset data into a pandas dataframe
        
    Arguments:
        wavs {[type]} -- [description]
    
    Keyword Arguments:
        _filetype {str} -- [description] (default: {"wav"})
    
    Returns:
        [type] -- [description]
    """
    wav_df = pd.DataFrame(
        columns=["monkey1", "monkey2", "date", "date_idx", _filetype + "_loc"]
    )
    for _wav in wavs:
        if _wav.stem[0] == ".":
            continue
        monkey1 = None
        date = None
        monkey2 = None
        date_idx = None

        wav_split = _wav.stem.split("_")
        if len(wav_split) == 3:
            monkey1, monkey2, date = wav_split
        elif len(wav_split) == 4:
            monkey1, monkey2, date, date_idx = wav_split
        elif len(wav_split) == 1:
            if len(_wav.stem.split(".")) == 3:
                monkey1, monkey2, date = _wav.stem.split(".")
            elif len(_wav.stem.split(".")) == 2:
                monkey1, date_idx = _wav.stem.split(".")
            elif len(_wav.stem.split(".")) == 4:
                monkey1, date_idx, date, _ = _wav.stem.split(".")
            elif len(re.findall("[A-Z][^A-Z]*", _wav.stem)) == 2:
                monkey1, date_idx = re.findall("[A-Z][^A-Z]*", _wav.stem)
            else:
                continue

        wav_df.loc[len(wav_df)] = [monkey1, monkey2, date, date_idx, _wav]
    return wav_df


def parse_marmoset_calls(row, callers=["monkey1_data", "monkey2_data"]):
    """ Parses a .mat file of marmoset vocalizations into a dataframe
        
    Arguments:
        row {[type]} -- [description]
    
    Keyword Arguments:
        callers {list} -- [description] (default: {["monkey1_data", "monkey2_data"]})
    
    Returns:
        [type] -- [description]
    """
    # load the annotations
    annotations = loadmat(row.mat_loc.as_posix())
    # create syllable_df
    syllable_df = pd.DataFrame(
        columns=[
            "indv",
            "partner",
            "date",
            "call_type",
            "wav_loc",
            "call_num",
            "pulse_n",
            "pulse_start",
            "pulse_end",
        ]
    )
    for caller in callers:
        # determine partner vs indv.
        indv = row.monkey1 if caller == "monkey1_data" else row.monkey2
        partner = row.monkey2 if caller == "monkey1_data" else row.monkey1
        for call_ix, call in enumerate(annotations[caller]):
            # this list goes [start1, end1, start2, end2]
            n_subcalls = int(len(call[1]) / 2)
            call_name = call[0][0]  # e.g. "phee"
            for call_sub in range(n_subcalls):
                subcall_start = call[1][call_sub * 2]
                subcall_end = call[1][(call_sub * 2) + 1]
                # if this call is too long, its probably a mistake
                if ((subcall_end - subcall_start) > 5) or (
                    (subcall_end - subcall_start) <= 0
                ):
                    continue
                syllable_df.loc[len(syllable_df)] = [
                    indv,
                    partner,
                    row.date,
                    call_name,
                    row.wav_loc,
                    call_ix,
                    call_sub,
                    subcall_start[0],
                    subcall_end[0],
                ]
    return syllable_df
