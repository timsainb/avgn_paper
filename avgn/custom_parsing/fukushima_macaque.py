from avgn.utils.audio import get_samplerate
import librosa
from avgn.utils.json import NoIndent, NoIndentEncoder
import json
import avgn
from avgn.utils.paths import DATA_DIR

DATASET_ID = "macaque_coo"


def generate_json(row, DT_ID):
    # get sr and duration
    sr = get_samplerate(row.wavloc.as_posix())
    wav_duration = librosa.get_duration(filename=row.wavloc)

    # create json
    json_dict = {}
    json_dict["common_name"] = "Macaque"
    json_dict["species"] = "Macaque mulatta"
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration
    json_dict["wav_loc"] = row.wavloc.as_posix()
    json_dict["idnum"] = row.idnum
    json_dict["samplerate_hz"] = sr
    json_dict["indvs"] = {
        row.indv: {
            "coos": {
                "start_times": NoIndent([0.0]),
                "end_times": NoIndent([wav_duration]),
            }
        }
    }
    json_out = (
        DATA_DIR
        / "processed"
        / DATASET_ID
        / DT_ID
        / "JSON"
        / (row.wavloc.stem + ".JSON")
    )
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
