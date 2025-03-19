import dataclasses as _dataclasses
import os as _os
import warnings as _warnings
from pathlib import Path as _Path

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
    def unigram_dir(self) -> _Path:
        return self.DATA_DIR / "datasets"/ "src"/"unigram"

    @property
    def model_dir(self) -> _Path:
        return self.DATA_DIR / "models"

    @property
    def result_dir(self) -> _Path:
        return self.DATA_DIR / "results"

    @property
    def script_dir(self) -> _Path:
        return self.DATA_DIR / "target_neuron_ablation"
    
    @property
    def config_dir(self) -> _Path:
        return self.DATA_DIR / "target_neuron_ablation"/ "experiments" / "conf"



#######################################################
# Instance of Settings
PATH = _MyPathSettings()