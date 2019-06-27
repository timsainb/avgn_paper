from avgn.utils.audio import get_samplerate
import librosa
from avgn.utils.json import NoIndentEncoder
import json
import avgn
from avgn.utils.paths import DATA_DIR

common_names = {
    'Great blue heron': 'Ardea herodias',
    'Indigo bunting': 'Passerina cyanea',
    'American crow': 'Corvus brachyrhynchos',
    'American yellow warbler': 'Setophaga petechia',
    'Blue jay': 'Cyanocitta cristata',
    'House finch': 'Haemorhous mexicanus',
    'Chipping sparrow': 'Spizella passerina',
    'Song sparrow': 'Melospiza melodia',
    'Common yellowthroat': 'Geothlypis trichas',
    'Cedar waxwing': 'Bombycilla cedrorum',
    'Marsh wren': 'Cistothorus palustris',
}

def generate_json(row, DT_ID):
    species = row.species.lstrip().capitalize()
    DATASET_ID = "NA_BIRDS_" + species.lower().replace(" ", "_")
    
    # sample rate and duration
    sr = get_samplerate(row.wavloc.as_posix())
    wav_duration = librosa.get_duration(filename=row.wavloc)
    
    # make json dictionary
    json_dict = {}
    json_dict["indvs"] = {
        "UNK": {"syllables": {"start_times": [0], "end_times": [wav_duration]}}
    }
    # add species
    json_dict["species"] = species
    json_dict["common_name"] = common_names[species]

    # add wav number
    json_dict["wav_num"] = int(row.wavnum)
    # add wav location
    json_dict["wav_loc"] = row.wavloc.as_posix()

    # rate and length
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration
    
    # dump json
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    # save information
    json_name = species.lower().replace(" ", "_")+'_'+str(row.wavnum).zfill(4)
    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (json_name + ".JSON")
    )

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))


