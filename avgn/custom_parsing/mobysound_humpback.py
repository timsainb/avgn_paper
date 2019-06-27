from avgn.utils.audio import get_samplerate
from avgn.utils.json import NoIndent, NoIndentEncoder
import numpy as np
from avgn.utils.paths import DATA_DIR
import librosa
from datetime import datetime
import pandas as pd
import avgn
import json

DATASET_ID = 'mobysound_humpback_whale'

def load_labs(labels):
    all_labels = []
    for label_file in labels:
        label_df = pd.DataFrame(
            [line.split() for line in open(label_file, "r")],
            columns=["start_time", "end_time", "low_freq", "high_freq", "SNR"],
        )
        label_df['file'] = label_file.stem
        all_labels.append(label_df)
    all_labels = pd.concat(all_labels).reset_index()
    for lab in ['start_time', 'end_time', 'low_freq', 'high_freq', 'SNR']:
        all_labels[lab] = all_labels[lab].values.astype('float32')
    return all_labels

def find_longest_nonvocal_stretch(file_df, wav_duration):
    """ An ugly function to find the longest stretch of nonvocal behavior in a syllable dataframe
    """
    ## find the longest stretch of non-vocal behavior in this wav
    max_break = np.argmax(file_df.start_time.values[1:] - file_df.end_time.values[:-1])
    noise_end_time = file_df.start_time.values[1:][max_break]
    noise_start_time = file_df.end_time.values[:-1][max_break]
    start_noise = file_df.start_time.values[0]
    end_noise = wav_duration - file_df.end_time.values[-1]
    noise_lens = np.array([noise_end_time - noise_start_time, start_noise, end_noise])
    noise_start_ends = np.array(
        [
            [noise_start_time, noise_end_time],
            [0, start_noise],
            [file_df.end_time.values[-1], wav_duration],
        ]
    )
    noise_start, noise_end = noise_start_ends[np.argmax(noise_lens)]
    return noise_start, noise_end

def generate_noise_and_json(bout_number, fn, DT_ID, wavloc, file_df):
    # location of wav
    #wavloc = np.array(wavs)[np.array([i.stem for i in wavs]) == fn][0]
    # wav time
    wavdate = datetime.strptime(fn, "%y%m%d-%H%M")
    wav_date = wavdate.strftime("%Y-%m-%d_%H-%M-%S")
    # wav samplerate and duration
    sr = get_samplerate(wavloc.as_posix())
    wav_duration = librosa.get_duration(filename=wavloc)
    # df of syllables in file
    #file_df = label_df[label_df.file == fn].sort_values(by="start_time")

    ## find the longest stretch of non-vocal behavior in this wav
    noise_start, noise_end = find_longest_nonvocal_stretch(file_df, wav_duration)
    bout_start_string = avgn.utils.general.seconds_to_str(noise_start)

    # determine save locations
    noise_out = (
        DATA_DIR
        / "processed"
        / DATASET_ID
        / DT_ID
        / "NOISE"
        / (fn + "__" + bout_start_string + ".WAV")
    )

    json_out = DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (fn + ".JSON")

    # wav general information
    json_dict = {}
    json_dict["bout_number"] = bout_number
    json_dict["species"] = "Megaptera novaengliae"
    json_dict["common_name"] = "Humpback whale"
    json_dict["datetime"] = wav_date
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration
    json_dict["wav_loc"] = wavloc.as_posix()
    json_dict["noise_loc"] = noise_out.as_posix()
    json_dict["indvs"] = {
        "UNK": {
            "syllables": {
                "start_times": NoIndent(
                    list(file_df.start_time.values.astype("float"))
                ),
                "end_times": NoIndent(list(file_df.end_time.astype("float"))),
                "high_freq": NoIndent(list(file_df.high_freq.astype("float"))),
                "low_freq": NoIndent(list(file_df.low_freq.astype("float"))),
                "SNR": NoIndent(list(file_df.SNR.astype("float"))),
            }
        }
    }

    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    # save wav file
    noise_wav, sr = librosa.load(
        wavloc, sr=None, mono=True, offset=noise_start, duration=noise_end - noise_start
    )
    avgn.utils.paths.ensure_dir(noise_out)
    librosa.output.write_wav(noise_out, y=noise_wav, sr=sr, norm=True)

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
