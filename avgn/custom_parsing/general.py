import numpy as np
import librosa


def extract_noise_clip(wav_loc, bout_start, bout_end, voc_starts, voc_ends, min_noise_clip_size_s, max_noise_clip_size_s):
    """ Given a list of vocalization start and end times, and the start and end time 
    of the current bout, look for noise near the current bout to segment out as a 
    noise clip. First try before, then after. 
        
    Arguments:
        wav_loc {[type]} -- location of wav file
        bout_start {[type]} -- start of current bout (seconds)
        bout_end {[type]} -- end of current bout (seconds)
        voc_starts {[type]} -- Times of vocalization starts (seconds)
        voc_ends {[type]} -- Times of vocalization ends (seconds)
        hparams {[type]} -- hparams with min_noise_clip_size_s and max_noise_clip_size_s
    
    Returns:
        [type] -- [description]
    """

    def extract_noise_pre():
        # try to get a noise clip from the time preceding this clip
        if bout_start > min_noise_clip_size_s:
            # get time of preceding pulses
            td = bout_start - voc_ends
            td = td[td > 0]
            # if there is anything within this timeframe, this timeframe is unusable
            if not np.any(td < min_noise_clip_size_s):
                # get times for noise clip
                noise_start = bout_start - np.min(
                    list(td + 1) + [max_noise_clip_size_s]
                )
                noise_end = bout_start

                # load the clip
                noise_clip, sr = librosa.load(
                    wav_loc,
                    mono=True,
                    sr=None,
                    offset=noise_start,
                    duration=noise_end - noise_start,
                )
                return noise_clip, sr
        return None, None

    def extract_noise_post():
        # try to get noise clip from end of file
        wav_duration = librosa.get_duration(filename=wav_loc)
        if wav_duration - bout_end > min_noise_clip_size_s:
            td = voc_starts - bout_end
            td = td[td > 0]
            if not np.any(td < min_noise_clip_size_s):
                # get times for noise clip
                noise_start = bout_end
                noise_end = bout_end + np.min(
                    list(td - min_noise_clip_size_s / 2)
                    + [max_noise_clip_size_s]
                )
                # load the clip
                noise_clip, sr = librosa.load(
                    wav_loc,
                    mono=True,
                    sr=None,
                    offset=noise_start,
                    duration=noise_end - noise_start,
                )
                return noise_clip, sr
        return None, None

    noise_clip, sr = extract_noise_pre()
    if noise_clip is None:
        noise_clip, sr = extract_noise_post()
    return noise_clip, sr
