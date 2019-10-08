import librosa
from avgn.utils.json import NoIndentEncoder
from datetime import datetime
from avgn.utils.paths import DATA_DIR, ensure_dir
from avgn.utils.audio import get_samplerate
import json

DATASET_ID = "european_starling_gentner"


def generate_json(row, DT_ID):
    datet = datetime.strptime(row.wavdate, "%Y-%m-%d_%H-%M-%S-%f")
    datestr = datet.strftime("%Y-%m-%d_%H-%M-%S")
    sr = get_samplerate(row.wavloc.as_posix())
    wav_duration = librosa.get_duration(filename=row.wavloc.as_posix())
    # general json info
    # make json dictionary
    json_dict = {}
    json_dict["species"] = "European starling"
    json_dict["common_name"] = "Sturnus vulgaris"
    json_dict["indvs"] = {row.indv: {}}
    json_dict["datetime"] = datestr
    # rate and length
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration
    json_dict["wav_loc"] = row.wavloc.as_posix()

    # generate json
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    json_out = (
        DATA_DIR
        / "processed"
        / DATASET_ID
        / DT_ID
        / "JSON"
        / (row.wavloc.stem + ".JSON")
    )

    # save json
    ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
