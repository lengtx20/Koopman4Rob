from pydantic import BaseModel, PositiveInt
from typing import List, Optional, Literal
from pathlib import Path


NAME = "blip2"
cam_keys = ["/env_camera/color/image_raw"]
feature_suffix = "features_proj"


class TestConfig(BaseModel):
    """Configuration for testing.
    Args:
        save_results: bool, whether to save the test results.
        show_plot: bool, whether to show plots of the results.
        rollout_steps: PositiveInt, number of steps for rollout prediction (>=1).
    """

    save_results: bool = True
    show_plot: bool = False
    rollout_steps: PositiveInt = 1


class Config(BaseModel):
    """Main configuration for the Koopman model.
    Args:
        data_dir: str, path to the data directory.
        model_dir: Path, path to the model directory.
        fisher_path: Optional[str], path to the Fisher information matrix file.
        mode: Literal["test", "train"], mode of operation.
        img_features_keys: List[str], list of keys for image features.
        test: TestConfig, configuration for testing.
    """

    data_dir: str
    model_dir: Path = Path(f"logs/{NAME}_150")
    fisher_path: Optional[str] = None
    mode: Literal["test", "train"] = "test"
    img_features_keys: List[str] = [f"{cam}/{feature_suffix}" for cam in cam_keys]
    test: TestConfig = TestConfig()
