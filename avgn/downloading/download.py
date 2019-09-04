import warnings
from tqdm.autonotebook import tqdm
import requests
import math
from avgn.utils.paths import ensure_dir
from pathlib2 import PosixPath, Path


def download_tqdm(url, output_location, block_size=1024):
    """ Download a file using requests and tqdm
    https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    """

    # ensure that the file is in the correct spot
    if type(output_location) != PosixPath:
        output_location = Path(output_location)
    if output_location.is_dir():
        output_location = output_location / Path(url).name

    if output_location.exists():
        warnings.warn("File {} already exists".format(output_location))
        return
    # make directory inf needed
    if not output_location.parent.exists():
        ensure_dir(output_location.parent)
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get("content-length", 0))

    wrote = 0
    with open(output_location, "wb") as f:
        for data in tqdm(
            r.iter_content(block_size),
            total=math.ceil(total_size // block_size),
            unit="KB",
            unit_scale=True,
        ):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")

