import librosa
from avgn.utils.json import NoIndent, NoIndentEncoder
import json
import avgn
from avgn.utils.paths import DATA_DIR
from scipy.io import loadmat
import pandas as pd
from tqdm.autonotebook import tqdm
from datetime import datetime

DATASET_ID = "bengalese_finch_sober"


def generate_json_wav(row, CBINLIST, cbin_names, DT_ID):
    """ generates a json and WAV for bengalese finch data in MAT and CBIN format
    """
    cbin_file = np.array(CBINLIST)[cbin_names == row.wavname][0]
    bout_wav, rate = load_cbin(cbin_file.as_posix())

    # general json info
    # make json dictionary
    json_dict = {}
    json_dict["species"] = "Lonchura striata domestica"
    json_dict["common_name"] = "Bengalese finch"
    json_dict["indvs"] = {
        row.bird: {
            "syllables": {
                "start_times": NoIndent(list(row.start_times)),
                "end_times": NoIndent(list(row.end_times)),
                "labels": NoIndent(list(row.syllables)),
            }
        }
    }
    wav_date = row.stime.strftime("%Y-%m-%d_%H-%M-%S")
    json_dict["datetime"] = wav_date
    # rate and length
    json_dict["samplerate_hz"] = rate
    json_dict["length_s"] = len(bout_wav) / rate

    wav_stem = row.wavname[:-5]

    # output locations
    wav_out = DATA_DIR / "processed" / DATASET_ID / DT_ID / "WAV" / (wav_stem + ".WAV")
    json_dict["wav_loc"] = wav_out.as_posix()
    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON")
    )

    # encode json
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    # save wav file
    avgn.utils.paths.ensure_dir(wav_out)
    librosa.output.write_wav(
        wav_out, y=bout_wav.astype("float32"), sr=int(rate), norm=True
    )

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))


def parse_song_df(MATLIST):
    song_df = pd.DataFrame(
        columns=[
            "bird",
            "species",
            "stime",
            "syllables",
            "start_times",
            "end_times",
            "bout_duration",
            "syll_lens",
            "day",
            "wavname",
            "rate",
        ]
    )
    for label_loc in tqdm(MATLIST):
        mat = loadmat(label_loc)
        loc_time = datetime.strptime(
            "_".join(label_loc.stem.split(".")[0].split("_")[-2:]), "%d%m%y_%H%M"
        )
        indv = label_loc.stem.split("_")[0]
        syll_lens = np.squeeze(mat["offsets"] - mat["onsets"]) / 1000
        labels = list(np.array(mat["labels"]).flatten()[0])
        start_times = np.array(mat["onsets"]).flatten() / 1000.0
        end_times = np.array(mat["offsets"]).flatten() / 1000.0
        bout_duration = (mat["offsets"][-1][0] - mat["onsets"][0][0]) / 1000
        # mat['bout_duration']/1000
        song_df.loc[len(song_df)] = [
            indv,
            "BF",
            loc_time,
            labels,
            start_times,
            end_times,
            bout_duration,
            syll_lens,
            loc_time.strftime("%d/%m/%y"),
            str(mat["fname"]).split("\\")[-1][:-2],
            int(mat["Fs"]),
        ]
    song_df["NumNote"] = [len(i) for i in song_df.syllables.values]

    song_df["rec_num"] = None
    song_df = song_df.reset_index()
    for bird in np.unique(song_df.bird):
        song_idxs = song_df[song_df.bird.values == bird].sort_values(by="stime").index
        for idxi, idx in enumerate(song_idxs):
            song_df.set_value(idx, "rec_num", idxi)
    song_df["day"] = [
        pd.to_datetime(str(i)).strftime("%Y-%m-%d") for i in song_df.stime.values
    ]
    return song_df


############################## FROM HVC ##################################
# this code for loading the cbin audio files was acquired from https://github.com/NickleDave/hybrid-vocal-classifier which is liscensed as BSD- 3

license = """BSD 3-Clause License

Copyright (c) 2016, David Nicholson
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

import numpy as np


def readrecf(filename):
    """
    reads .rec files output by EvTAF
    """

    rec_dict = {}
    with open(filename, "r") as recfile:
        line_tmp = ""
        while 1:
            if line_tmp == "":
                line = recfile.readline()
            else:
                line = line_tmp
                line_tmp = ""

            if line == "":  # if End Of File
                break
            elif line == "\n":  # if blank line
                continue
            elif "Catch" in line:
                ind = line.find("=")
                rec_dict["iscatch"] = line[ind + 1 :]
            elif "Chans" in line:
                ind = line.find("=")
                rec_dict["num_channels"] = int(line[ind + 1 :])
            elif "ADFREQ" in line:
                ind = line.find("=")
                try:
                    rec_dict["sample_freq"] = int(line[ind + 1 :])
                except ValueError:
                    rec_dict["sample_freq"] = float(line[ind + 1 :])
            elif "Samples" in line:
                ind = line.find("=")
                rec_dict["num_samples"] = int(line[ind + 1 :])
            elif "T After" in line:
                ind = line.find("=")
                rec_dict["time_after"] = float(line[ind + 1 :])
            elif "T Before" in line:
                ind = line.find("=")
                rec_dict["time before"] = float(line[ind + 1 :])
            elif "Output Sound File" in line:
                ind = line.find("=")
                rec_dict["outfile"] = line[ind + 1 :]
            elif "Thresholds" in line:
                th_list = []
                while 1:
                    line = recfile.readline()
                    if line == "":
                        break
                    try:
                        th_list.append(float(line))
                    except ValueError:  # because we reached next section
                        line_tmp = line
                        break
                rec_dict["thresholds"] = th_list
                if line == "":
                    break
            elif "Feedback information" in line:
                fb_dict = {}
                while 1:
                    line = recfile.readline()
                    if line == "":
                        break
                    elif line == "\n":
                        continue
                    ind = line.find("msec")
                    time = float(line[: ind - 1])
                    ind = line.find(":")
                    fb_type = line[ind + 2 :]
                    fb_dict[time] = fb_type
                rec_dict["feedback_info"] = fb_dict
                if line == "":
                    break
            elif "File created" in line:
                header = [line]
                for counter in range(4):
                    line = recfile.readline()
                    header.append(line)
                rec_dict["header"] = header
    return rec_dict


def load_cbin(filename, channel=0):
    """
    loads .cbin files output by EvTAF. 
    
    arguments
    ---------
    filename : string
    channel : integer
        default is 0
    returns
    -------
    data : numpy array
        1-d vector of 16-bit signed integers
    sample_freq : integer
        sampling frequency in Hz. Typically 32000.
    """

    # .cbin files are big endian, 16 bit signed int, hence dtype=">i2" below
    data = np.fromfile(filename, dtype=">i2")
    recfile = filename[:-5] + ".rec"
    rec_dict = readrecf(recfile)
    data = data[channel :: rec_dict["num_channels"]]  # step by number of channels
    sample_freq = rec_dict["sample_freq"]
    return data, sample_freq

