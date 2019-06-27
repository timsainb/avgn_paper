from avgn.utils.audio import get_samplerate
import librosa
from avgn.utils.json import NoIndentEncoder
import json
import avgn
from avgn.utils.paths import DATA_DIR
import re

common_names = {
    'PicusViridis': 'European green woodpecker',
    'DryocopusMartius': 'Black woodpecker',
    'DendrocoposMedius': 'Middle spotted woodpecker',
    'JynxTorquilla': 'Eurasian wryneck',
    'DendrocoposLeucotos': 'Whick-backed woodpecker',
    'DendrocoposMinor': 'Lesser spotted woodpecker',
    'DendrocoposMajor': 'Great spotted woodpecker' 
}

def generate_json(row, DT_ID):
    common_name = common_names[row.species]
    species = ' '.join(re.findall('[A-Z][^A-Z]*', row.species)).capitalize()
    # wav info
    sr = get_samplerate(row.wavloc.as_posix())
    wav_duration = librosa.get_duration(filename=row.wavloc)
    fn = row.wavloc.stem
    
    DATASET_ID = 'woodpecker_' + species.lower().replace(' ', '_')
    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (fn + ".JSON")
    )
    
    # make json dictionary
    json_dict = {}
    json_dict["indvs"] = {
        "UNK": {}
    }
    # add species
    json_dict["species"] = species
    json_dict["common_name"] = common_name
    json_dict["wav_loc"] = row.wavloc.as_posix()
    json_dict["sound_type"] = row.call_type
    json_dict["origin"] = row.origin
    # rate and length
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration
    
    # dump json
    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))