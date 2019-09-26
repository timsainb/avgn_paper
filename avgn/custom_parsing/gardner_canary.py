import librosa
from avgn.utils.json import NoIndent, NoIndentEncoder
import pandas as pd
from datetime import datetime
from praatio import tgio
from avgn.utils.paths import DATA_DIR, ensure_dir
from avgn.utils.audio import get_samplerate
import json

DATASET_ID = "canary"


def get_phrases(tg, WAVLIST, wav_stems):
    phrase_df = pd.DataFrame(
        columns=[
            "indv",
            "rendition",
            "datetime",
            "wavloc",
            "tgloc",
            "phrase_num",
            "phrase_start",
            "phrase_end",
            "phrase_label",
        ]
    )
    indv, rendition, year, month, day, hour, minute = tg.stem.split("_")
    dt = "-".join([year, month, day, hour, minute])
    dt = datetime.strptime(dt, "%Y-%m-%d-%H-%M")
    wf = WAVLIST[wav_stems == tg.stem][0]
    textgrid = tgio.openTextgrid(fnFullPath=tg)
    tier = textgrid.tierDict["syllables"].entryList
    for inti, interval in enumerate(tier):

        phrase_df.loc[len(phrase_df)] = [
            indv,
            rendition,
            dt,
            wf,
            tg,
            inti,
            interval.start,
            interval.end,
            interval.label,
        ]
    return phrase_df


def gen_wav_json(wf, wav_df, DT_ID, save_wav=False):
    """ generates a JSON of segmental iformation from the wav_df row
    
    if the flag save_wav is set to true, also generates a WAV file

    Arguments:
        wf {[type]} -- [description]
        wav_df {[type]} -- [description]
        DT_ID {[type]} -- [description]
    
    Keyword Arguments:
        save_wav {bool} -- [description] (default: {False})
    """

    wav_stem = wf.stem

    # output locations
    if save_wav:
        # load wav file
        bout_wav, sr = librosa.load(wf, mono=True, sr=None)

        wav_out = (
            DATA_DIR / "processed" / DATASET_ID / DT_ID / "WAV" / (wav_stem + ".WAV")
        )
        bout_duration = len(bout_wav) / sr
        # save wav file
        ensure_dir(wav_out)
        librosa.output.write_wav(wav_out, y=bout_wav, sr=sr, norm=True)
    else:
        sr = get_samplerate(wav_df.iloc[0].wavloc.as_posix())
        wav_out = wav_df.iloc[0].wavloc
        bout_duration = librosa.get_duration(filename=wav_df.iloc[0].wavloc.as_posix())

    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON")
    )

    # create json dictionary
    indv = wav_df.iloc[0].indv
    json_dict = {}
    json_dict["indvs"] = {indv: {"phrases": {}}}
    json_dict["rendition"] = wav_df.iloc[0].rendition
    json_dict["datetime"] = wav_df.iloc[0].datetime.strftime("%Y-%m-%d_%H-%M-%S")
    json_dict["original_wav"] = wav_df.iloc[0].wavloc.as_posix()
    json_dict["samplerate_hz"] = sr
    json_dict["indvs"][indv]["phrases"]["start_times"] = NoIndent(
        list(wav_df.phrase_start.values)
    )
    json_dict["indvs"][indv]["phrases"]["end_times"] = NoIndent(
        list(wav_df.phrase_end.values)
    )
    json_dict["indvs"][indv]["phrases"]["labels"] = NoIndent(
        list(wav_df.phrase_label.values)
    )
    json_dict["wav_loc"] = wav_out.as_posix()
    json_dict["length_s"] = bout_duration
    json_dict["species"] = "Serinus canaria forma domestica"
    json_dict["common_name"] = "Domestic canary"

    # generate json
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    # save json
    ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
