import librosa
from avgn.utils.json import  NoIndentEncoder
from datetime import datetime
import json
import avgn
from avgn.utils.paths import DATA_DIR


DATASET_ID = 'hagen_common_spadefoot'

def generate_wav_json(row, DT_ID, mp3_path):
	wavdate = datetime.strptime(row.recording_date, '%d.%m.%Y')
	wav_date = wavdate.strftime("%Y-%m-%d_%H-%M-%S")

	bout_wav, sr = librosa.load(mp3_path)

	wav_duration = len(bout_wav)/sr

	fn = row.filename_ext.split('.')[0]

	# output locations
	wav_out = (
	    DATA_DIR
	    / "processed"
	    / DATASET_ID
	    / DT_ID
	    / "WAV"
	    / (fn + ".WAV")
	)
	    
	json_out = (
	    DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (fn + ".JSON")
	)

	# make json dictionary
	json_dict = {}
	json_dict["indvs"] = {
	    "UNK": {"syllables": {"start_times": [0], "end_times": [wav_duration]}}
	}
	# add species
	json_dict["species"] = 'Common spadefoot'
	json_dict["common_name"] = 'Pelobates fuscus'

	json_dict["original_wav"] = mp3_path.as_posix()

	json_dict["sound_type"] = row.sound_type
	json_dict["age"] = row.age
	json_dict["locality"] = row.locality
	json_dict["administrative_area"] = row.administrative_area
	json_dict["country"] = row.country
	json_dict["state"] = row.state
	json_dict["datetime"] = wav_date

	# add wav location
	json_dict["wav_loc"] = wav_out.as_posix()
	# rate and length
	json_dict["samplerate_hz"] = sr
	json_dict["length_s"] = wav_duration


	# dump json
	json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

	# save json
	avgn.utils.paths.ensure_dir(json_out.as_posix())
	print(json_txt, file=open(json_out.as_posix(), "w"))


	# save wav file
	avgn.utils.paths.ensure_dir(wav_out)
	librosa.output.write_wav(wav_out, y=bout_wav, sr=sr, norm=True)

