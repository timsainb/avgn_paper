from avgn.utils.paths import DATA_DIR
import avgn
from avgn.utils.json import NoIndent, NoIndentEncoder
import json
import numpy as np
import librosa
import xml.etree.ElementTree
from avgn.utils.audio import get_samplerate
import pandas as pd
from tqdm.autonotebook import tqdm

DATASET_ID = "koumura_bengalese_finch"


def Koumura_Okanoya_parser(bird_xml_locs, wav_list):
    """ parses XML from Koumura_Okanoya data format
    """
    song_df = pd.DataFrame(
        columns=[
            "bird",
            "WaveFileName",
            "Position",
            "Length",
            "NumNote",
            "NotePositions",
            "NoteLengths",
            "NoteLabels",
        ]
    )
    for bird_loc in tqdm(bird_xml_locs):
        bird_xml = xml.etree.ElementTree.parse(bird_loc).getroot()
        bird = bird_loc.parent.stem
        for element in tqdm(bird_xml.getchildren(), leave=False):
            if element.tag == "Sequence":
                notePositions = []
                noteLengths = []
                noteLabels = []
                for seq_element in element.getchildren():
                    if seq_element.tag == "Position":
                        position = seq_element.text
                    elif seq_element.tag == "Length":
                        length = seq_element.text
                    elif seq_element.tag == "WaveFileName":
                        WaveFileName = seq_element.text
                    elif seq_element.tag == "NumNote":
                        NumNote = seq_element.text
                    elif seq_element.tag == "Note":
                        for note_element in seq_element.getchildren():
                            if note_element.tag == "Label":
                                noteLabels.append(note_element.text)
                            elif note_element.tag == "Position":
                                notePositions.append(note_element.text)
                            elif note_element.tag == "Length":
                                noteLengths.append(note_element.text)
                song_df.loc[len(song_df)] = [
                    bird,
                    WaveFileName,
                    position,
                    length,
                    NumNote,
                    notePositions,
                    noteLengths,
                    noteLabels,
                ]

    return song_df


def generate_json(DSLOC, DT_ID, bird, wfn, wfn_df):

    # wav location
    wav_loc = DSLOC / bird / "Wave" / wfn

    # wav info
    sr = get_samplerate(wav_loc.as_posix())
    wav_duration = librosa.get_duration(filename=wav_loc)

    # make json dictionary
    json_dict = {}
    # add species
    json_dict["species"] = "Lonchura striata domestica"
    json_dict["common_name"] = "Bengalese finch"
    json_dict["wav_loc"] = wav_loc.as_posix()
    # rate and length
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration

    # make a dataframe of wav info
    # wfn_df = bird_df[bird_df.WaveFileName == wfn]
    seq_df = pd.DataFrame(
        (
            [
                [
                    list(np.repeat(sequence_num, len(row.NotePositions))),
                    list(row.NoteLabels),
                    np.array(
                        (np.array(row.NotePositions).astype("int") + int(row.Position))
                        / sr
                    ).astype("float64"),
                    np.array(
                        (
                            np.array(row.NotePositions).astype("int")
                            + np.array(row.NoteLengths).astype("int")
                            + int(row.Position)
                        )
                        / sr
                    ).astype("float64"),
                ]
                for sequence_num, (idx, row) in enumerate(wfn_df.iterrows())
            ]
        ),
        columns=["sequence_num", "labels", "start_times", "end_times"],
    )
    # add syllable information
    json_dict["indvs"] = {
        bird: {
            "notes": {
                "start_times": NoIndent(
                    list(np.concatenate(seq_df.start_times.values))
                ),
                "end_times": NoIndent(list(np.concatenate(seq_df.end_times.values))),
                "labels": NoIndent(list(np.concatenate(seq_df.labels.values))),
                "sequence_num": NoIndent(
                    [int(i) for i in np.concatenate(seq_df.sequence_num.values)]
                ),
            }
        }
    }

    # dump json
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    wav_stem = bird + "_" + wfn.split(".")[0]
    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON")
    )

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
