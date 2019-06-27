from avgn.utils.paths import DATA_DIR
import avgn
from avgn.utils.json import NoIndentEncoder
import json
import librosa
from avgn.utils.audio import get_samplerate

species_dict_common = {
    'CC': 'cicada',
    'KA': 'catydid',
    'CR': 'cricket',
    'MQ': 'mosquito',
    'WA': 'wasp',
    'MI': 'midge',
    'BE': 'bee',
    'FF': 'fruitfly',
    'BT': 'beetle'
}
species_dict = {
    'CC': 'Cicadoidea sp.',
    'KA': 'Tettigoniidae sp.',
    'CR': 'Gryllidae sp.',
    'MQ': 'Culicidae sp.',
    'WA': '	Hymenoptera sp.',
    'MI': 'midge',
    'BE': 'Nematocera sp.',
    'FF': 'Drosophila sp.',
    'BT': 'Coleoptera sp.'
}

def generate_json(row, DT_ID):

    # wav info
    sr = get_samplerate(row.wavloc.as_posix())
    wav_duration = librosa.get_duration(filename=row.wavloc)

    # make json dictionary
    json_dict = {}
    # add species
    json_dict["species_id"] = row.species

    json_dict["species"] = species_dict[row.species_group]
    json_dict["common_name"] = species_dict_common[row.species_group]
    json_dict["wav_loc"] = row.wavloc.as_posix()

    # rate and length
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration

    # add syllable information
    json_dict["indvs"] = {
        row.species: {}
    }

    DATASET_ID = 'insect_dataset_'+ species_dict_common[row.species_group]

    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    wav_stem = row.wavloc.stem

    json_out = (
        DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON")
    )

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))
