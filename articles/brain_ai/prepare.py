"""Download the data behind the post's brain maps into articles/brain_ai/data, where it is also kept in the repository.

    LanA: Lipkin et al. (2022), a probabilistic atlas of the language network from 804 people on fsaverage,
    figshare doi:10.6084/m9.figshare.20425209, CC BY 4.0

    uv run --group articles python articles/brain_ai/prepare.py
"""

import io
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).parent / "data"
LANA = "https://ndownloader.figshare.com/files/36524916"  # FS Atlas.zip


def main():
    DATA.mkdir(exist_ok=True)
    if not (DATA / "LH_LanA_n804.nii.gz").exists():
        with urllib.request.urlopen(LANA) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        for name in archive.namelist():
            if name.endswith("_LanA_n804.nii.gz"):
                (DATA / Path(name).name).write_bytes(archive.read(name))
    print(sorted(p.name for p in DATA.iterdir()))


if __name__ == "__main__":
    main()
