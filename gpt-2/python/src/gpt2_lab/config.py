"""Small, explicit experiment settings; no model is allocated when loading them."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import re


def positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 32
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 32

    def __post_init__(self) -> None:
        for field in fields(self):
            positive_integer(field.name, getattr(self, field.name))
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def gpt2_124m(cls) -> "ModelConfig":
        return cls(50257, 1024, 12, 12, 768)

    @classmethod
    def tutorial1_tiny(cls) -> "ModelConfig":
        # Tutorial 1's final Tiny Shakespeare run: 65 characters, two 16-wide blocks.
        return cls(65, 16, 2, 2, 16)


@dataclass(frozen=True)
class DatasetConfig:
    path: str
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("dataset.path must be a nonempty file path")
        if self.encoding != "utf-8":
            raise ValueError("The scaffold currently supports UTF-8 text datasets")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 1337
    device: str = "cpu"
    batch_size: int = 2
    max_steps: int = 20
    learning_rate: float = 0.0003

    def __post_init__(self) -> None:
        positive_integer("batch_size", self.batch_size)
        positive_integer("max_steps", self.max_steps)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.device not in ("cpu", "cuda", "mps", "auto"):
            raise ValueError("device must be cpu, cuda, mps, or auto")
        if (type(self.learning_rate) not in (float, int)
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("learning_rate must be finite and positive")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", self.name):
            raise ValueError("name must use 1-64 lowercase letters, digits, or hyphens")
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Only experiment schema_version 1 is supported")


@dataclass(frozen=True)
class LoadedExperiment:
    config: ExperimentConfig
    config_path: Path
    config_sha256: str

    @property
    def dataset_path(self) -> Path:
        # Anchor paths to the configuration, so the launch directory cannot change the dataset.
        return (self.config_path.parent / Path(self.config.dataset.path).expanduser()).resolve()

    def plan(self) -> dict:
        return {
            "status": "plan-only: training is not implemented yet",
            "experiment": asdict(self.config),
            "config_path": str(self.config_path),
            "dataset_path": str(self.dataset_path),
            "dataset_exists": self.dataset_path.is_file(),
            "head_size": self.config.model.head_size,
        }


def section(config_type: type, values: dict):
    if not isinstance(values, dict):
        raise ValueError(f"{config_type.__name__} must be a JSON object")
    unknown = set(values) - {field.name for field in fields(config_type)}
    if unknown:
        raise ValueError(f"Unknown {config_type.__name__} fields: {', '.join(sorted(unknown))}")
    try:
        return config_type(**values)
    except TypeError as error:
        raise ValueError(f"Invalid {config_type.__name__}: {error}") from error


def load_experiment(path: str | Path) -> LoadedExperiment:
    source = Path(path).expanduser().resolve()
    content = source.read_bytes()
    values = json.loads(content.decode("utf-8-sig"))
    if not isinstance(values, dict):
        raise ValueError("An experiment must be a JSON object")
    values = dict(values)
    for name, config_type in (("model", ModelConfig), ("dataset", DatasetConfig),
                              ("training", TrainingConfig)):
        if name not in values:
            raise ValueError(f"Missing experiment section: {name}")
        values[name] = section(config_type, values[name])
    config = section(ExperimentConfig, values)
    return LoadedExperiment(config, source, hashlib.sha256(content).hexdigest())
