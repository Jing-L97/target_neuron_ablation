import dataclasses as _dataclasses
import os as _os
import warnings as _warnings
from pathlib import Path as _Path

import torch

#######################################################
# precision setting
dtype_dict = {
    torch.float32: ["EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-410m-deduped"],
    torch.float16: ["EleutherAI/pythia-6.9B-deduped", "EleutherAI/pythia-2.8B-deduped", "EleutherAI/pythia-1B-deduped"],
}


def get_dtype(model_name=None, dtype_dict=dtype_dict, default=torch.float32):
    """Get precision (dtype) for a given model name using a reverse search."""
    for dtype, model_list in dtype_dict.items():
        if model_name in model_list:
            return dtype
    return default


#######################################################
# directory setting


def cache_dir() -> _Path:
    """Return a directory to use as cache."""
    cache_path = _Path(_os.environ.get("CACHE_DIR", _Path.home() / ".cache" / __package__))
    if not cache_path.is_dir():
        cache_path.mkdir(exist_ok=True, parents=True)
    return cache_path


def _assert_dir(dir_location: _Path) -> None:
    """Check if directory exists & throw warning if it doesn't."""
    if not dir_location.is_dir():
        _warnings.warn(
            f"Using non-existent directory: {dir_location}\nCheck your settings & env variables.",
            stacklevel=1,
        )


#######################################################
# Path Settings


@_dataclasses.dataclass
class _MyPathSettings:
    DATA_DIR: _Path = _Path(_os.environ.get("DATA_DIR", "data/"))
    COML_SERVERS: tuple = tuple({"oberon", "oberon2", "habilis", *[f"puck{i}" for i in range(1, 7)]})
    KNOWN_HOSTS: tuple[str, ...] = (*COML_SERVERS, "MacBook-Pro-de-jliu")

    def __post_init__(self) -> None:
        if "DATA_DIR" not in _os.environ:
            self.DATA_DIR = _Path("/scratch2/jliu/Generative_replay/neuron")

        if not self.DATA_DIR.is_dir():
            _warnings.warn(
                f"Provided DATA_DIR: {self.DATA_DIR} does not exist.\n"
                "You either need to run the code in one of the predifined servers.\n"
                "OR provide a valid DATA_DIR env variable.",
                stacklevel=1,
            )

    @property
    def dataset_root(self) -> _Path:
        _assert_dir(self.DATA_DIR / "datasets")
        return self.DATA_DIR / "datasets"

    @property
    def script_dir(self) -> _Path:
        return self.DATA_DIR / "target_neuron_ablation"

    @property
    def model_dir(self) -> _Path:
        return self.DATA_DIR / "models"

    @property
    def result_dir(self) -> _Path:
        return self.DATA_DIR / "results"

    @property
    def unigram_dir(self) -> _Path:
        return self.DATA_DIR / "datasets" / "src" / "unigram"

    @property
    def context_dir(self) -> _Path:
        return self.DATA_DIR / "datasets" / "context"

    @property
    def surprisal_dir(self) -> _Path:
        return self.DATA_DIR / "results" / "surprisal"

    @property
    def ablation_dir(self) -> _Path:
        return self.DATA_DIR / "results" / "ablations"

    @property
    def neuron_dir(self) -> _Path:
        return self.DATA_DIR / "results" / "selection"

    @property
    def direction_dir(self) -> _Path:
        return self.DATA_DIR / "results" / "geometry"

    @property
    def classify_dir(self) -> _Path:
        return self.DATA_DIR / "results" / "classify"


#######################################################
# Instance of Settings

PATH = _MyPathSettings()
