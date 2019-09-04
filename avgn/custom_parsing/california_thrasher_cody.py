import librosa
from avgn.utils.json import NoIndent, NoIndentEncoder
import pandas as pd
from datetime import datetime
from praatio import tgio
from avgn.utils.paths import DATA_DIR, ensure_dir
from avgn.utils.audio import get_samplerate
import json
from datetime import time as dtt

DATASET_ID = 'california_thrasher_cody'

def generate_json(wavfile, DT_ID, song_db):
    indv = wavfile.parent.parent.stem
    dt = datetime.strptime(wavfile.stem, "%Y-%m-%d_%H-%M-%S-%f")
    datestring = dt.strftime("%Y-%m-%d")

    row=song_db[
                (song_db.SubjectName == indv)
                & (song_db.recording_date == datestring)
                & (song_db.recording_time == dt.time())
            ].iloc[0]

    # make json dictionary
    json_dict = {}
    for key in dict(row).keys():
        if type(row[key]) == pd._libs.tslibs.timestamps.Timestamp:
            json_dict[key] = row[key].strftime("%Y-%m-%d_%H-%M-%S")
        elif type(row[key]) == dtt:
            json_dict[key] = row[key].strftime("%H:%M:%S")
        elif type(row[key]) == pd._libs.tslibs.nattype.NaTType:
            continue
        else:
            json_dict[key] = row[key]


    json_dict["species"] = "Toxostoma redivivum"
    json_dict["common_name"] = "California thrasher"
    json_dict["datetime"] = datestring

    sr = get_samplerate(wavfile.as_posix())
    wav_duration = librosa.get_duration(filename=wavfile.as_posix())

    # rate and length
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration
    json_dict["wav_loc"] = wavfile.as_posix()

    tg = wavfile.parent.parent / "TextGrids" / (wavfile.stem + ".TextGrid")

    textgrid = tgio.openTextgrid(fnFullPath=tg)

    tierlist = textgrid.tierDict[textgrid.tierNameList[0]].entryList
    start_times = [i.start for i in tierlist]
    end_times = [i.end for i in tierlist]
    labels = [i.label for i in tierlist]

    json_dict["indvs"] = {
        indv: {
            "syllables": {
                "start_times": NoIndent(start_times),
                "end_times": NoIndent(end_times),
                "labels": NoIndent(labels),
            }
        }
    }

    # generate json
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)


    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wavfile.stem + ".JSON")
    )

    # save json
    ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))