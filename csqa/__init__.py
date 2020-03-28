import pathlib
from pathlib import PosixPath
import os

from csqa.version import VERSION as __version__


DATA_DIR = (
    os.getenv('CSQA_DATA') or
    os.path.join(PosixPath(__file__).absolute().parents[1].as_posix(), 'data')
)
