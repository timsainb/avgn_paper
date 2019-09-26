import struct
from avgn.utils.paths import DATA_DIR
from avgn.utils.json import NoIndent, NoIndentEncoder
import json
import numpy as np
from datetime import datetime
import avgn
import librosa

DATASET_ID = "swamp_sparrow"


def string2int16(string):
    byte_array = bytes.fromhex(string)
    count = int(len(byte_array) / 2)
    return struct.unpack("h" * count, byte_array)


def annotate_bouts(
    row, songdata_row, individual_row, wav_elements, wav_syllables, DT_ID
):
    """Grabs annotation information for swampsparrow and creates JSON labels and saves wav
    
    [description]
    
    Arguments:
        row {[type]} -- [description]
        songdata_row {[type]} -- [description]
        individual_row {[type]} -- [description]
        wav_elements {[type]} -- [description]
        wav_syllables {[type]} -- [description]
        DT_ID {[type]} -- [description]
    """

    if type(row.WAV) == float:
        if np.isnan(row.WAV):
            return

    # recording time
    recording_time = datetime.fromtimestamp(row.TIME / 1000.0).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    # output locations
    wav_stem = songdata_row.NAME.split(".")[0]
    wav_out = DATA_DIR / "processed" / DATASET_ID / DT_ID / "WAV" / (wav_stem + ".WAV")
    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON")
    )

    # wav general information
    json_dict = {}

    json_dict["datetime"] = recording_time
    json_dict["samplerate_hz"] = row.SAMPLERATE
    json_dict["indvs"] = {individual_row.NAME: {}}
    json_dict["MAXFREQ"] = float(songdata_row.MAXFREQ)
    json_dict["RECORDER"] = songdata_row.RECORDER
    json_dict["species"] = "Melospiza georgiana"
    json_dict["common_name"] = individual_row.SPECID
    json_dict["POPID"] = individual_row.POPID
    json_dict["LOCDESC"] = individual_row.LOCDESC
    json_dict["GRIDTYPE"] = individual_row.GRIDTYPE
    json_dict["GRIDX"] = individual_row.GRIDX
    json_dict["GRIDY"] = individual_row.GRIDY
    json_dict["SEX"] = individual_row.SEX
    json_dict["AGE"] = individual_row.AGE
    json_dict["RANK"] = individual_row.RANK
    json_dict["wav_loc"] = wav_out.as_posix()

    # load the wav
    wavdata = string2int16(row.WAV)
    sr = int(row.SAMPLERATE)

    # populate with syllable information

    json_dict["indvs"][individual_row.NAME]["syllables"] = {}

    syllable_start_times = []
    syllable_end_times = []
    for idx, syllable_row in wav_syllables[1:].iterrows():
        syllable_start_times.append(
            (syllable_row.STARTTIME) / 1000
        )  # * row.SAMPLERATE)
        syllable_end_times.append((syllable_row.ENDTIME) / 1000)  # * row.SAMPLERATE)

    json_dict["indvs"][individual_row.NAME]["syllables"]["start_time"] = NoIndent(
        syllable_start_times
    )
    json_dict["indvs"][individual_row.NAME]["syllables"]["end_time"] = NoIndent(
        syllable_end_times
    )

    # populate with element information
    json_dict["indvs"][individual_row.NAME]["elements"] = {}
    element_start_times = []
    element_end_times = []
    element_syllable_num = []
    element_pos_in_syllable = []

    for idx, element_row in wav_elements.iterrows():
        # timings of element
        element_start = (
            element_row.STARTTIME * element_row.TIMESTEP
        ) / 1000  # * row.SAMPLERATE
        element_end = element_start + (element_row.TIMELENGTH / 1000)
        element_start_times.append(element_start)
        element_end_times.append(element_end)
        element_middle = (element_start + element_end) / 2
        # which syllable does this element belong to
        syllable_num = np.where(
            (np.array(syllable_start_times) <= element_middle)
            & (np.array(syllable_end_times) >= element_middle)
        )[0]
        if len(syllable_num) > 0:
            syllable_num = syllable_num[0]
        else:
            syllable_num = -1

        element_syllable_num.append(syllable_num)

        # what number element within the sylalble is this
        if len(element_pos_in_syllable) > 0:
            if syllable_num == element_syllable_num[-2]:
                # this syllable is the same
                element_pos_in_syllable.append(element_pos_in_syllable[-1] + 1)
            else:
                element_pos_in_syllable.append(0)
        else:
            element_pos_in_syllable.append(0)

    # add information about elements
    json_dict["indvs"][individual_row.NAME]["elements"]["pos_in_syllable"] = NoIndent(
        element_pos_in_syllable
    )
    json_dict["indvs"][individual_row.NAME]["elements"]["syllable"] = NoIndent(
        [int(i) for i in element_syllable_num]
    )
    json_dict["indvs"][individual_row.NAME]["elements"]["start_times"] = NoIndent(
        element_start_times
    )
    json_dict["indvs"][individual_row.NAME]["elements"]["end_times"] = NoIndent(
        element_end_times
    )

    # dump
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    # save wav file
    avgn.utils.paths.ensure_dir(wav_out)
    librosa.output.write_wav(
        wav_out, y=np.array(wavdata).astype("float32"), sr=sr, norm=True
    )

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
