from avgn.utils.json import NoIndentEncoder
import librosa
import json
import avgn
from avgn.utils.paths import DATA_DIR

DATASET_ID = 'zebra_finch_gardner'

def generate_json_wav_noise(indv, wav_num, song, nonsong, sr, DT_ID):

    wav_duration = len(song) / sr

    wav_stem = indv + "_" + str(wav_num).zfill(4)

    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON")
    )
    noise_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "NOISE" / (wav_stem + ".WAV")
    )
    wav_out = DATA_DIR / "processed" / DATASET_ID / DT_ID / "WAV" / (wav_stem + ".WAV")

    # make json dictionary
    json_dict = {}
    # add species
    json_dict["species"] = "Taeniopygia guttata"
    json_dict["common_name"] = "Zebra finch"
    json_dict["wav_loc"] = wav_out.as_posix()
    json_dict["noise_loc"] = noise_out.as_posix()

    # rate and length
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration
    json_dict["wav_num"] = wav_num

    # add syllable information
    json_dict["indvs"] = {
        indv: {"motifs": {"start_times": [0.0], "end_times": [wav_duration]}}
    }

    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    # save wav file
    print(wav_out)
    avgn.utils.paths.ensure_dir(wav_out)
    librosa.output.write_wav(wav_out, y=song, sr=int(sr), norm=True)

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))

    # save noise
    avgn.utils.paths.ensure_dir(noise_out)
    librosa.output.write_wav(noise_out, y=nonsong, sr=int(sr), norm=True)