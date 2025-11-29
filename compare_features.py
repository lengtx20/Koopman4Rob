from data.blip2_feature_extractor import Blip2ImageFeatureExtractor
from pathlib import Path
import cv2
import torch


extractor = Blip2ImageFeatureExtractor(model_path="pretrained_models/blip2-itm-vit-g")
prompt = "The end effector of the robotic arm tries to get close to the QR code attached to the cabinet"
extractor.load_model()

for date in ["20251129_212239"]:
    for rollout in [0]:
        images = {
            "batch": f"logs/infer/{date}/{rollout}/batch",
            "live": f"logs/infer/{date}/{rollout}",
        }
        names = ["env_camera.color.image_raw", "follow_camera.color.image_raw"]
        for name in names:
            features = []
            for key, dir in images.items():
                img = cv2.imread(Path(dir) / name / "0.png")[:, :, ::-1].copy()
                feature = extractor.process_image(img, prompt)["features_proj"].squeeze(
                    0
                )
                features.append(feature)
            print(torch.norm(features[0]) - torch.norm(features[1]).tolist())
            print(name, torch.norm(features[0] - features[1]).tolist())
