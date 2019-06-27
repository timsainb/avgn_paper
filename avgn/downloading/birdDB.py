import avgn
from avgn.utils.paths import DATA_DIR
import pandas as pd
import xlrd
from urllib.error import HTTPError
from datetime import timedelta
import urllib.request


def openBirdDB_df():
    song_db = pd.read_excel(DATA_DIR / "BIRD_DB.xls")
    mainData_book = xlrd.open_workbook(DATA_DIR / "BIRD_DB.xls", formatting_info=True)
    mainData_sheet = mainData_book.sheet_by_index(0)
    song_urls = [
        ""
        if mainData_sheet.hyperlink_map.get((i, 11)) == None
        else mainData_sheet.hyperlink_map.get((i, 11)).url_or_path
        for i in range(mainData_sheet.nrows)
    ]
    song_db["Audio_file"] = song_urls[1:]
    song_db = song_db[1:]
    song_db["file_stem"] = [
        i.split("/")[-1].split(".")[0] for i in song_db["Audio_file"].values
    ]
    return song_db


def downloadBirdDB(row, bird_db_loc):
    wav = row["Audio_file"]
    text_grid = row["Textgrid_file"]
    # track_name = row["TrackName"]
    subject_id = row["SubjectName"]
    species = row["Species_short_name"]
    recording_time = row["recording_date"] + timedelta(
        hours=row["recording_time"].hour,
        minutes=row["recording_time"].minute,
        seconds=row["recording_time"].second,
    )
    # PREP SAVE LOCATION
    recording_time_string = recording_time.strftime("%Y-%m-%d_%H-%M-%S-%f")
    wav_location = (
        bird_db_loc / species / subject_id / "wavs" / (recording_time_string + ".wav")
    )
    grid_location = (
        bird_db_loc
        / species
        / subject_id
        / "TextGrids"
        / (recording_time_string + ".TextGrid")
    )

    avgn.utils.paths.ensure_dir(wav_location.as_posix())
    avgn.utils.paths.ensure_dir(grid_location.as_posix())

    # save wav
    if not wav_location.is_file():
        try:
            urllib.request.urlretrieve(wav, wav_location)
        except HTTPError:
            print("Could not retrieve " + wav)

    # save textgrid
    if not grid_location.is_file():
        try:
            urllib.request.urlretrieve(
                "http://taylor0.biology.ucla.edu/birdDBQuery/Files/" + text_grid,
                grid_location,
            )
        except HTTPError:
            print(
                "Could not retrieve "
                + "http://taylor0.biology.ucla.edu/birdDBQuery/Files/"
                + text_grid
            )
