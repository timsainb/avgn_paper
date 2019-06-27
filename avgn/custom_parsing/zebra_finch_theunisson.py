from avgn.utils.paths import DATA_DIR
import avgn
from avgn.utils.json import NoIndentEncoder
import json
import librosa
from avgn.utils.audio import get_samplerate
from tqdm.autonotebook import tqdm
import pandas as pd
from datetime import datetime

DATASET_ID = "zebra_finch_theunisson"

call_dict = {
    "Ag": "Wsst or aggressive call",
    "Be": "Begging calls",
    "DC": "Distance call",
    "Di": "Distress call",
    "LT": "Long Tonal call",
    "Ne": "Nest call",
    "So": "Song",
    "Te": "Tet call",
    "Th": "Thuk call",
    "Tu": "Tuck call",
    "Wh": "Whine call",
    "WC": "Unknown",
}


def parse_wavlist(WAVLIST):
    wav_df = pd.DataFrame(
        columns=[
            "indv",
            "age",
            "recordingdate",
            "vocalization_type",
            "voc_type_full",
            "voc_num",
            "wav_loc",
        ]
    )
    for wl in tqdm(WAVLIST):
        if wl.stem[: len("Unknown")] == "Unknown":
            continue
        indv = wl.stem[:10]
        anno = wl.stem[11:].split("-")
        if len(anno) == 3:
            wdate, call_type_full, call_num = anno
        elif len(anno) == 2:
            wdate, call_type_full = anno[0].split("_")
            call_num = anno[1]
        elif len(anno) == 4:
            wdate = anno[0]
            call_type_full = "".join(anno[1:3])
            call_num = anno[3]
        wdate = datetime.strptime(wdate[:6], "%y%m%d")
        age = wl.parent.stem
        wav_df.loc[len(wav_df)] = [
            indv,
            age,
            wdate,
            call_type_full[:2],
            call_type_full,
            call_num,
            wl,
        ]
    return wav_df


def generate_json(row, DT_ID):

    # wav info
    sr = get_samplerate(row.wav_loc.as_posix())
    wav_duration = librosa.get_duration(filename=row.wav_loc)

    # make json dictionary
    json_dict = {}
    # add species
    json_dict["species"] = "Taeniopygia guttata"
    json_dict["common_name"] = "Zebra finch"
    json_dict["wav_loc"] = row.wav_loc.as_posix()

    # rate and length
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration
    json_dict["wav_num"] = row.voc_num

    json_dict["vocalization_type"] = row.vocalization_type
    json_dict["voc_type_full"] = row.voc_type_full
    json_dict["voc_type_def"] = call_dict[row.vocalization_type]
    json_dict["age"] = row.age
    json_dict["datetime"] = row.recordingdate.strftime("%Y-%m-%d_%H-%M-%S")

    # add syllable information
    json_dict["indvs"] = {
        row.indv: {"elements": {"start_times": [0.0], "end_times": [wav_duration]}}
    }

    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    wav_stem = row.wav_loc.stem

    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON")
    )

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
