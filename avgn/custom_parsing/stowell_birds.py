import pandas as pd
from avgn.utils.audio import get_samplerate
import librosa
from avgn.utils.json import NoIndentEncoder
import json
import avgn
from avgn.utils.paths import DATA_DIR


def parse_csv(csvrow, DSLOC):
    wav_df = pd.DataFrame(
        columns=[
            "species",
            "year",
            "fgbg",
            "trntst",
            "indv",
            "cutted",
            "groundx",
            "wavnum",
            "wavloc",
        ]
    )
    csv = pd.read_csv(csvrow.csvloc)
    for idx, wavrow in csv.iterrows():
        cutted, bgx, indv, wavnum = wavrow.wavfilename[:-4].split("_")
        wf = list(DSLOC.glob("wav/*/" + wavrow.wavfilename))
        if len(wf) > 0:
            wf = wf[0]
        else:
            print("No wav available, skipping")
        wav_df.loc[len(wav_df)] = [
            csvrow.species,
            csvrow.withinacross,
            csvrow.fgbg,
            csvrow.traintest,
            indv,
            cutted,
            bgx,
            wavnum,
            wf,
        ]
    return wav_df


def generate_json(row, DT_ID, noise_indv_df):
    """ generate a json from available wav information for stowell dataset
    """
    DATASET_ID = "stowell_" + row.species

    sr = get_samplerate(row.wavloc.as_posix())
    wav_duration = librosa.get_duration(filename=row.wavloc)

    # make json dictionary
    json_dict = {}
    json_dict["indvs"] = {row.indv: {}}
    # add species
    json_dict["species"] = row.species

    species = {
    	"chiffchaff": "Phylloscopus collybita",
    	"littleowl" : "Athene noctua",
    	"pipit" : "Anthus trivialis",
    }
    json_dict["species"] = species[row.species]
    json_dict["common_name"] = row.species

    # add year information
    json_dict["year"] = row.year
    # add train/test split
    json_dict["train"] = row.trntst
    # add wav number
    json_dict["wav_num"] = int(row.wavnum)
    # add wav location
    json_dict["wav_loc"] = row.wavloc.as_posix()

    # rate and length
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration
    
    # get noise loc
    noise_indv_df = noise_indv_df[
        (noise_indv_df.species == row.species)]
    noise_indv_df = noise_indv_df[
        (noise_indv_df.year == row.year)]
    noise_indv_df = noise_indv_df[
        (noise_indv_df.groundx == row.groundx)]
    noise_indv_df = noise_indv_df[
        (noise_indv_df.fgbg == 'bg')]
      
    if len(noise_indv_df[noise_indv_df.wavnum == row.wavnum]) > 0:
        noise_loc = (
            noise_indv_df[noise_indv_df.wavnum == row.wavnum].iloc[0].wavloc.as_posix()
        )
    else:
        if len(noise_indv_df) > 0:
            noise_loc = noise_indv_df.iloc[0].wavloc.as_posix()
        else:
            noise_loc = ''

    return
    json_dict["noise_loc"] = noise_loc

    # dump json
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    # save information
    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (row.wavloc.stem + ".JSON")
    )

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))