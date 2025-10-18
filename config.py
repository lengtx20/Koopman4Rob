from pydantic import BaseModel
from typing import List, Optional

NAME = "blip2"
cam_keys = ["/env_camera/color/image_raw"]
feature_suffix = "features_proj"


class Config(BaseModel):
    data_dir: str
    model_dir: str = f"logs/{NAME}_150"
    fisher_path: Optional[str] = None
    mode: str = "test"
    img_features_keys: List[str] = [f"{cam}/{feature_suffix}" for cam in cam_keys]
