import re
from scipy.io import loadmat
import pandas as pd
from avgn.utils.json import NoIndent, NoIndentEncoder
from datetime import datetime
import librosa
import json
import avgn
from avgn.utils.paths import DATA_DIR

DATASET_ID = "marmoset"


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


def segment_wav_into_bouts(wav_df, hparams):
    """ Segments the wav_df full of segmental information into individual bouts
    """
    # populate a list of dataframes corresponding to each bout
    bout_dfs = []
    # first bout starts at first voc
    bout_start = wav_df.iloc[0].pulse_start
    for ri, (idx, row) in enumerate(wav_df.iterrows()):

        # if this is the last voc, it should be the end of the bout
        if ri == len(wav_df) - 1:
            bout_end = row.pulse_end
        # if there is not a gap greater than bout_segmentation_min_s after this voc its part of the same voc
        if ri == len(wav_df) - 1:
            bout_end = row.pulse_end
        else:
            if (
                wav_df.iloc[ri + 1].pulse_start - row.pulse_end
                > hparams.bout_segmentation_min_s
            ):
                bout_end = row.pulse_end
            else:
                continue

        # create a dataframe of only the bout
        bout_df = wav_df[
            (wav_df.pulse_start >= bout_start) & (wav_df.pulse_end <= bout_end)
        ]
        bout_dfs.append(bout_df)

        # set next bout start
        if ri < len(wav_df) - 1:
            bout_start = wav_df.iloc[ri + 1].pulse_start

    return bout_dfs





def annotate_bouts(DT_ID, bout_number, wav_df, bout_df, hparams):
    """ segments parsed bouts and annotates as json
    """

    bout_start = bout_df.pulse_start.values[0]
    bout_end = bout_df.pulse_end.values[-1]
    # Ensure padding does not start before WAV starts
    bout_pad_start = hparams.bout_pad_s
    if bout_start - hparams.bout_pad_s < 0:
        bout_pad_start = hparams.bout_pad_s - bout_start

    # load the wav at the relevant times + padding if possible
    clip_duration = (bout_end + hparams.bout_pad_s) - (bout_start - bout_pad_start)
    bout_wav, sr = librosa.load(
        bout_df.iloc[0].wav_loc,
        mono=True,
        sr=None,
        offset=bout_start - bout_pad_start,
        duration=clip_duration,
    )
    # extract a noise clip
    if hparams.get_noise_clip:
        bout_noise, noise_sr = avgn.custom_parsing.general.extract_noise_clip(
            bout_df.iloc[0].wav_loc,
            bout_start,
            bout_end,
            wav_df.pulse_start.values,
            wav_df.pulse_end.values,
            hparams.min_noise_clip_size_s,
            hparams.max_noise_clip_size_s,
        )
    else:
        bout_noise = None
        noise_sr = None

    # get time of bout relative to wav
    time_in_wav = bout_start - bout_pad_start
    bout_start_string = avgn.utils.general.seconds_to_str(time_in_wav)
    wav_stem = bout_df.iloc[0].wav_loc.stem

    # output locations
    wav_out = (
        DATA_DIR
        / "processed"
        / DATASET_ID
        / DT_ID
        / "WAV"
        / (wav_stem + "__" + bout_start_string + ".WAV")
    )
    json_out = (
        DATA_DIR
        / "processed"
        / DATASET_ID
        / DT_ID
        / "JSON"
        / (wav_stem + "__" + bout_start_string + ".JSON")
    )

    noise_out = (
        DATA_DIR
        / "processed"
        / DATASET_ID
        / DT_ID
        / "NOISE"
        / (wav_stem + "__" + bout_start_string + ".WAV")
    )

    bout_duration = len(bout_wav) / sr
    # generate the json for the bout
    wavdate = datetime.strptime(bout_df.date.values[0], "%d%m%y")
    wav_date = wavdate.strftime("%Y-%m-%d_%H-%M-%S")

    # wav general information
    json_dict = {}
    json_dict["species"] = "Callithrix jacchus"
    json_dict["common_name"] = "Common marmoset"
    json_dict["bout_number"] = bout_number
    json_dict["datetime"] = wav_date
    json_dict["samplerate_hz"] = sr
    json_dict["original_wav"] = bout_df.wav_loc.values[0].as_posix()
    json_dict["length_s"] = bout_duration
    json_dict["time_relative_to_original_wav"] = bout_start - bout_pad_start
    json_dict["wav_loc"] = wav_out.as_posix()
    json_dict["noise_loc"] = noise_out.as_posix()
    json_dict["indvs"] = {}

    # individual specific information
    for indv in bout_df.indv.unique():
        json_dict["indvs"][indv] = {}
        indv_df = bout_df[bout_df.indv == indv].sort_values(by="pulse_start")
        json_dict["indvs"][indv]["partner"] = indv_df.partner.values[0]
        json_dict["indvs"][indv]["calls"] = {
            "start_times": NoIndent(
                list(indv_df.pulse_start.values - bout_start + bout_pad_start)
            ),
            "end_times": NoIndent(
                list(indv_df.pulse_end.values - bout_start + bout_pad_start)
            ),
            "labels": NoIndent(list(indv_df.call_type.values)),
            "call_num": NoIndent(list(indv_df.call_num.values)),
            "pulse_num": NoIndent(list(indv_df.pulse_n.values)),
        }

    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    # save wav file
    avgn.utils.paths.ensure_dir(wav_out)
    librosa.output.write_wav(wav_out, y=bout_wav, sr=sr, norm=True)

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))

    # save noise file
    if hparams.get_noise_clip:

        avgn.utils.paths.ensure_dir(noise_out)
        if bout_noise is not None:
            librosa.output.write_wav(noise_out, y=bout_noise, sr=noise_sr, norm=True)
