import librosa
from avgn.utils.json import NoIndent, NoIndentEncoder
import pandas as pd
from datetime import datetime
from praatio import tgio
from avgn.utils.paths import DATA_DIR, ensure_dir
from avgn.utils.audio import get_samplerate
import json
from datetime import time as dtt

mouse_id_dict = {
    "A": "Aco59_2",
    "B": "Aco59_2",
    "C": "Can15-1",
    "D": "Can15-1",
    "E": "Can16-1",
    "F": "Can16-1",
    "G": "Can9-1",
    "H": "Can9-1",
    "I": "Aco65_1",
    "L": "Can3_1",
}

species_dict = {
    "rat": "Rattus norvegicus domesticus",
    "gerbil": "Meriones unguiculatus",
    "mouse": "Mus musculus",
}


def generate_json(row, DT_ID):

    wav = row.wavloc

    cond = wav.parent.stem.split("_")
    if len(cond) == 2:
        common_name, condition = cond
    else:
        common_name = cond[0]
        condition = None

    if common_name == "mouse":
        if condition == "C57BL":
            data_id = wav.stem.split("_")[0]
            indv_id = mouse_id_dict[data_id]
        elif condition == "BALBc":
            indv_id = wav.stem.split("-")[0]
    elif common_name == "rat":
        indv_id = wav.stem.split("_")[-2]
    elif common_name == "gerbil":
        indv_id = wav.stem

    # wav info
    sr = get_samplerate(row.wavloc.as_posix())
    wav_duration = librosa.get_duration(filename=row.wavloc)
    species = species_dict[common_name]

    # make json dictionary
    json_dict = {}
    # add species
    json_dict["condition"] = condition
    json_dict["species"] = species
    json_dict["common_name"] = common_name
    json_dict["wav_loc"] = row.wavloc.as_posix()

    # rate and length
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration

    # get syllable start and end times
    csv = row.wavloc.parent / (row.wavloc.stem + ".csv")
    voc_df = pd.read_csv(csv, header=None)[[0, 1]]
    voc_df.columns = ["start_time", "end_time"]

    # add syllable information
    json_dict["indvs"] = {
        indv_id: {
            "syllables": {
                "start_times": NoIndent(list(voc_df.start_time.values)),
                "end_times": NoIndent(list(voc_df.end_time.values)),
            }
        }
    }

    DATASET_ID = "tachibana_" + common_name

    # dump
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)
    wav_stem = row.wavloc.stem

    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON")
    )
    wav_out = DATA_DIR / "processed" / DATASET_ID / DT_ID / "WAV" / (wav_stem + ".WAV")
    print(json_out)
    # save json
    ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
