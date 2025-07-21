import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pickle


class EncodeImage:
    """
    图像编码器类，使用预训练的ResNet模型提取图像特征
    """

    def __init__(self, model_name: str = "resnet34", device: str = "auto"):
        """
        初始化图像编码器

        Args:
            model_name: 使用的模型名称，支持 "resnet34", "resnet50", "resnet18"
            device: 设备类型，"auto" 自动选择，"cpu" 或 "cuda"
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_name)
        self.transform = self._setup_transform()

    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self, model_name: str) -> nn.Module:
        """加载预训练模型"""
        model: nn.Module = getattr(models, model_name)(
            weights=getattr(
                models, f"{model_name.replace('resnet', 'ResNet')}_Weights"
            ).DEFAULT
        )

        model.fc = nn.Identity()  # 移除最后的全连接层，保留特征提取部分
        model.to(self.device)
        model.eval()
        return model

    def _setup_transform(self) -> transforms.Compose:
        """设置图像预处理变换"""
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess_image(self, image_array: np.ndarray) -> torch.Tensor:
        """
        处理任意维度的图像数组，自动移除所有size=1的维度，最终保留3D (H, W, 3)

        Args:
            image_array: 输入图像数组

        Returns:
            处理后的图像张量
        """
        image_array = np.squeeze(image_array)  # 自动移除所有size=1的维度

        if image_array.ndim > 3:
            # 如果仍大于3维，尝试取第一个元素（针对批量图像的情况，如(2, H, W, 3)取第0个）
            image_array = image_array[0]  # 取第一个元素，将4维→3维
            # 再次检查维度（防止取元素后仍不符合）
            assert image_array.ndim == 3, (
                f"处理后仍为{image_array.ndim}D，shape: {image_array.shape}"
            )

        # 确保最终是3D数组
        assert image_array.ndim == 3, (
            f"期望3D数组，处理后为{image_array.ndim}D，shape: {image_array.shape}"
        )

        # 确保通道数为3（RGB，最后一维为3）
        assert image_array.shape[-1] == 3, (
            f"期望最后一维为3个通道（RGB），实际为{image_array.shape[-1]}，shape: {image_array.shape}"
        )

        image_tensor = self.transform(image_array)
        image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
        return image_tensor.to(self.device, torch.float32)

    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        从图像张量中提取特征

        Args:
            image_tensor: 预处理后的图像张量

        Returns:
            提取的特征
        """
        with torch.no_grad():
            # 模型输出 `features` 的形状是 `[B, 512, 1, 1]`（ResNet-34 的特征图）
            features = self.model(image_tensor)
        # 将特征展平为 `[B, 512]`
        # 这里的 `B` 是批次大小，通常为1
        return features.view(features.size(0), -1)

    def encode_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        完整的图像编码流程：预处理 + 特征提取

        Args:
            image_array: 输入图像数组

        Returns:
            提取的特征数组
        """
        tensor = self.preprocess_image(image_array)
        features = self.extract_features(tensor)
        return features

    def process_data(self, data_path: str = None, data: list = None) -> tuple:
        """
        处理数据集，提取所有图像的特征

        Args:
            data_path: 数据文件路径（.pkl文件）
            data: 直接传入的数据列表

        Returns:
            (features_wrist, features_env_close) 两个视角的特征数组
        """
        print(f"使用设备: {self.device}")

        # 加载数据
        if data_path:
            with open(data_path, "rb") as f:
                data = pickle.load(f)
        elif data is None:
            # 使用模拟数据
            data = [
                {
                    "observations": {
                        "wrist": np.random.rand(1, 224, 224, 3),  # 模拟数据
                        "env_close": np.random.rand(1, 224, 224, 3),  # 模拟数据
                    }
                }
            ]

        features_wrist = []
        features_env_close = []
        success_count = 0

        for i, episode in enumerate(data):
            try:
                obs = episode["observations"]
                wrist_arr = obs["wrist"]
                env_close_arr = obs["env_close"]
            except Exception as e:
                print(f"episode {i + 1} 处理失败: {str(e)}")
                continue

            print(f"\nepisode {i + 1} - 原始shape:")
            print(f"  wrist: {wrist_arr.shape}, env_close: {env_close_arr.shape}")

            # 提取特征
            feat_wrist = self.encode_image(wrist_arr)
            feat_env = self.encode_image(env_close_arr)

            # 保存特征
            features_wrist.append(feat_wrist)
            features_env_close.append(feat_env)
            success_count += 1
            print(
                f"episode {i + 1} 处理成功 {feat_wrist.squeeze().shape}（累计: {success_count}）"
            )

        print(f"\n处理完成，成功{success_count}/{len(data)}个episode")

        # 拼接特征（处理空列表情况）
        feat_wrist_np = (
            np.concatenate(features_wrist, axis=0) if features_wrist else np.array([])
        )
        feat_env_np = (
            np.concatenate(features_env_close, axis=0)
            if features_env_close
            else np.array([])
        )
        return feat_wrist_np, feat_env_np


def main(data_path: str):
    """保持向后兼容的主函数"""
    encoder = EncodeImage()
    return encoder.process_data(data_path)


if __name__ == "__main__":
    # 使用新的类接口
    encoder = EncodeImage(model_name="resnet34", device="auto")

    pkl_path = "data/open_drawer_20_demos_2025-07-09_21-48-22.pkl"
    features_wrist, features_env_close = encoder.process_data(pkl_path)

    # 保存特征向量为.npy文件
    np.save("features_wrist.npy", features_wrist)  # 保存腕部视角特征
    np.save("features_env_close.npy", features_env_close)  # 保存环境视角特征

    print("\n特征向量已保存为：")
    print("1. wrist视角: features_wrist.npy")
    print("2. env_close视角: features_env_close.npy")

    print("\n最终特征结果:")
    print(f"wrist特征: {features_wrist.shape if features_wrist.size > 0 else '空'}")
    print(
        f"env_close特征: {features_env_close.shape if features_env_close.size > 0 else '空'}"
    )
