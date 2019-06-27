import librosa
from avgn.utils.json import  NoIndentEncoder
from datetime import datetime
import json
import avgn
from avgn.utils.paths import DATA_DIR

DATASET_ID = 'budka_crex_crex'

def generate_wav_json(row, DT_ID, mp3_path):

    wavdate= datetime.strptime(row.recording_date + '_' +row.recording_time, "%d.%m.%Y_%H:%M")
    wav_date = wavdate.strftime("%Y-%m-%d_%H-%M-%S")

    # output locations
    wav_stem = row.filename_ext.split(".")[0]

    wav_out = DATA_DIR / "processed" / DATASET_ID / DT_ID / "WAV" / (wav_stem + ".WAV")
    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON")
    )

    # load mp3
    bout_wav, sr = librosa.load(mp3_path, sr=None, mono=True)
    wav_duration = len(bout_wav) / sr
    # indv info
    indv = row.filename.split("_")[-1]

    # general json info
    # make json dictionary
    json_dict = {}
    json_dict["indvs"] = {indv: {}}
    # add species
    json_dict["species"] = row.species
    json_dict["sound_type"] = row.sound_type
    json_dict["age"] = row.age
    json_dict["sex"] = row.sex
    json_dict["latitude"] = row.latitude
    json_dict["longitude"] = row.longitude
    json_dict["altitude"] = row.altitude
    json_dict["datetime"] = wav_date
    json_dict["species"] = "Crex crex"
    json_dict["common_name"] = "Corn crake"

    # add wav location
    json_dict["wav_loc"] = wav_out.as_posix()

    # rate and length
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration

    #
    print(json_dict)
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    # save wav file
    avgn.utils.paths.ensure_dir(wav_out)
    librosa.output.write_wav(wav_out, y=bout_wav, sr=sr, norm=True)

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))