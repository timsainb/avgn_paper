from avgn.utils.json import NoIndentEncoder
import librosa
import json
import avgn
from avgn.utils.paths import DATA_DIR

species_dict = {
    "Cuviers": {
        "species": "Ziphius cavirostris",
        "common_name": "Cuvier's beaked whale",
    },
    "Gervais": {
        "species": "Mesoplodon europaeus",
        "common_name": "Gervais's beaked whale",
    },
}


def generate_wav_json(row, rate, DT_ID):
    DATASET_ID = "hildebrand_" + species_dict[row.species]["common_name"].replace(
        " ", "_"
    ).replace("'", "")
    wav_date = row.time.strftime("%Y-%m-%d_%H-%M-%S.%f")
    # output locations
    wav_out = (
        DATA_DIR
        / "processed"
        / DATASET_ID
        / DT_ID
        / "WAV"
        / (wav_date.replace(".", "_") + ".WAV")
    )
    json_out = (
        DATA_DIR
        / "processed"
        / DATASET_ID
        / DT_ID
        / "JSON"
        / (wav_date.replace(".", "_") + ".JSON")
    )

    wav_data = row.MSN

    json_dict = {}
    json_dict["species"] = species_dict[row.species]["species"]
    json_dict["common_name"] = species_dict[row.species]["common_name"]
    json_dict["site"] = row.site
    json_dict["recording_num"] = row.rec_no
    json_dict["bout_i"] = row.bout_i

    json_dict["samplerate_hz"] = rate
    json_dict["length_s"] = rate / len(wav_data)

    json_dict["wav_loc"] = wav_out.as_posix()

    json_dict["indvs"] = {
        "UNK": {"clicks": {"start_times": [0.0], "end_times": [len(wav_data) / rate]}}
    }

    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)
    print(json_txt)

    # save wav file
    avgn.utils.paths.ensure_dir(wav_out)
    librosa.output.write_wav(wav_out, y=wav_data, sr=rate, norm=True)

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
