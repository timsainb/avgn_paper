from avgn.utils.json import  NoIndentEncoder
from datetime import datetime
from avgn.utils.audio import get_samplerate
import librosa
import json
from avgn.utils.paths import DATA_DIR
import avgn

DATASET_ID = 'castellucci_mouse_usv'

def generate_json(row, DT_ID):
    wavdate = datetime(year=int(row.year), day=int(row.day), month = int(row.month))
    wav_date = wavdate.strftime("%Y-%m-%d_%H-%M-%S")
    
    # wav samplerate and duration
    sr = get_samplerate(row.wav_loc.as_posix())
    wav_duration = librosa.get_duration(filename=row.wav_loc)
    
    # wav general information
    json_dict = {}
    json_dict["datetime"] = wav_date
    json_dict["samplerate_hz"] = sr
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration
    json_dict["species"] = "Mus musculus"
    json_dict["common_name"] = "House mouse"
    json_dict["wav_loc"] = row.wav_loc.as_posix()
    json_dict["age"] = row.AGE
    json_dict["FemaleMouse"] = row.FemaleMouse
    json_dict['call_type'] = row.SONG
    json_dict["weight"] = row.Weight
    json_dict["indvs"] = {
        row.indv: {}
    }
    
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    json_out = DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (row.wav_loc.stem + ".JSON")

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
