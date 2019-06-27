from avgn.utils.paths import DATA_DIR
import avgn
from avgn.utils.json import NoIndentEncoder
import json
import librosa
from avgn.utils.audio import get_samplerate


DATASET_ID = 'katahira_white_munia'

def generate_json(row, DT_ID):
	# wav info
	try:
		sr = get_samplerate(row.wavloc.as_posix())
	except Exception as e:
		print(row.wavloc.as_posix(), e)

	wav_duration = librosa.get_duration(filename=row.wavloc)

	# make json dictionary
	json_dict = {}
	# add species
	json_dict["species"] = "Lonchura striata"
	json_dict["common_name"] = "White rumped munia"
	json_dict["wav_loc"] = row.wavloc.as_posix()
	# rate and length
	json_dict["samplerate_hz"] = sr
	json_dict["length_s"] = wav_duration
	json_dict["wav_num"] = row.wav_num

	# add syllable information
	json_dict["indvs"] = {
	    row.indv: {
	        }
	    }

	# dump json
	json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

	wav_stem = row.indv + "_" + str(row.wav_num)
	json_out = (
	    DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON")
	)

	# save json
	avgn.utils.paths.ensure_dir(json_out.as_posix())
	print(json_txt, file=open(json_out.as_posix(), "w"))
