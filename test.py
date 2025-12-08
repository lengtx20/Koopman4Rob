from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from typing import Dict, Union


class Test(BaseModel):
    """A test class for demonstration purposes."""

    example_field: Union[Dict[str, dict], list] = {}
    """An example field."""


cfg = DictConfig({"example_key": {"sub_key": 123}})

print(OmegaConf.to_object(cfg))

test = Test(example_field=cfg)
print(test.example_field)
