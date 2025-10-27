import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class EWC:
    def __init__(self, model, data, loss_fn, device, threshold=None):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.threshold = threshold
        
        self.params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        # self.fisher = self.compute_fisher(data)
        self.fisher = None
        self.masks = {}

        if self.threshold is not None:
            self._generate_mask(self.threshold)

    def _generate_mask(self, threshold):
        for n in self.fisher:
            self.masks[n] = (self.fisher[n] >= threshold).float()

    def compute_fisher(self, data, batch_size=64):
        # Init dataloader for batch computation
        x_t = data[:, : self.model.state_dim]
        a_t = data[
            :, self.model.state_dim : self.model.state_dim + self.model.action_dim
        ]
        x_t1 = data[:, -self.model.state_dim :]

        # x_t_tensor = torch.from_numpy(x_t).float()
        # a_t_tensor = torch.from_numpy(a_t).float()
        # x_t1_tensor = torch.from_numpy(x_t1).float()

        dataset = TensorDataset(
            x_t.clone().detach(),
            a_t.clone().detach(),
            x_t1.clone().detach(),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Init fisher matrix
        fisher = {
            n: torch.zeros_like(p, device=self.device) for n, p in self.params.items()
        }
        self.model.eval()

        # tqdm包装迭代器，加进度条
        for x_t_batch, a_t_batch, x_t1_batch in tqdm(
            dataloader, desc="Computing Fisher"
        ):
            x_t_batch = x_t_batch.to(self.device)
            a_t_batch = a_t_batch.to(self.device)
            x_t1_batch = x_t1_batch.to(self.device)

            self.model.zero_grad()
            pred_x_t1 = self.model(x_t_batch, a_t_batch, get_action=False)
            loss = self.loss_fn(pred_x_t1, x_t1_batch)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * x_t_batch.shape[0]

        total_samples = len(data)
        for n in fisher:
            fisher[n] /= total_samples
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self.fisher[n] * (p - self.params[n]).pow(2)
                loss += _loss.sum()
        return loss

    def save(self, model_dir, task_id=1):
        """
        Save fisher weights and params to dir/ewc_task{task_id}.pt
        """
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, f"ewc_task{task_id}.pt")
        torch.save({"fisher_dict": self.fisher, "params": self.params}, save_path)
        print(f"[EWC] Saved fisher & params to {save_path}")


class Fisher_Analyzer:
    def __init__(self):
        self.file_path = None
        self.ckpt = None
        self.fisher_dict = None

    def load_fisher(self, file_path, task_id=None):
        self.ckpt = torch.load(file_path)

        self.fisher_dict = self.ckpt.get("fisher_dict", {})
        print("[INFO] fisher_dict type:", type(self.fisher_dict))
        print("[INFO] fisher_dict length:", len(self.fisher_dict))
        print("[INFO] fisher_dict keys:", self.fisher_dict.keys())

        if isinstance(self.fisher_dict, dict) and isinstance(
            list(self.fisher_dict.values())[0], dict
        ):
            self.fisher_dict = list(self.fisher_dict.values())[task_id - 1]

    def preprocess_K(self):
        self.off_diagonal = self.fisher_dict.get(
            "_koopman_propagator.off_diagonal", None
        )
        self.diagonal = self.fisher_dict.get("_koopman_propagator.diagonal", None)

        if self.off_diagonal is None or self.diagonal is None:
            print(
                "Missing required Fisher matrices (_koopman_propagator.off_diagonal or _koopman_propagator.diagonal)"
            )
            return None

        self.K_state_fisher = (
            torch.diag(self.diagonal) + self.off_diagonal - self.off_diagonal.t()
        )

    def summarize_fisher(self):
        print("\n[### Analyzing Structure - Fisher Matrix ###]")
        print(f"num of param matrix: {len(self.fisher_dict)}")
        for key in list(self.fisher_dict.keys()):
            print(f"- param name: {key}")
            print(f"  Fisher shape: {self.fisher_dict[key].shape}")

    def visualize_fisher(self, save_dir="models/ewc_0/figs", save_fig=False):
        os.makedirs(save_dir, exist_ok=True)
        print("\n[### Visualizing Fisher Matrix ###]")

        for i, (key, fisher_tensor) in enumerate(self.fisher_dict.items()):
            # tackle with koopman state matrix
            if key == "_koopman_propagator.off_diagonal":
                continue
            elif key == "_koopman_propagator.diagonal":
                self.preprocess_K()
                fisher_tensor = self.K_state_fisher

            fisher_cpu = fisher_tensor.detach().cpu().numpy()

            # check dimension of fisher_cpu
            if fisher_cpu.ndim == 1:
                fisher_cpu = fisher_cpu.reshape(1, -1)
            elif fisher_cpu.ndim != 2:
                print(f"Skipping {key}, Unable to visualize {fisher_cpu.ndim}D")
                continue

            # plot
            plt.figure(figsize=(6, 4))
            im = plt.imshow(fisher_cpu, cmap="viridis", aspect="auto")
            plt.colorbar(im)
            plt.title(f"Fisher: {key}")
            plt.tight_layout()

            if save_fig:
                file_name = os.path.join(
                    save_dir, f"fisher_{i}_{key.replace('.', '_')}.png"
                )
                plt.savefig(file_name)
                plt.close()
                print(f"- Saved to: {file_name}")
            else:
                plt.show()
                plt.close()

    def thresholded_visualize_fisher(
        self, threshold=100, save_dir="models/ewc_0/thresholded", save_fig=False
    ):
        print(
            f"\n[### Visualizing Thresholded Fisher Matrix (threshold={threshold}) ###]"
        )

        for i, (key, fisher_tensor) in enumerate(self.fisher_dict.items()):
            # tackle with koopman state matrix
            if key == "_koopman_propagator.off_diagonal":
                continue
            elif key == "_koopman_propagator.diagonal":
                self.preprocess_K()
                fisher_tensor = self.K_state_fisher

            fisher_processed = fisher_tensor.clone()
            fisher_processed[fisher_processed >= threshold] = 100.0
            fisher_processed[fisher_processed < threshold] = 0.0

            fisher_cpu = fisher_processed.detach().cpu().numpy()

            # check dimension
            if fisher_cpu.ndim == 1:
                fisher_cpu = fisher_cpu.reshape(1, -1)
            elif fisher_cpu.ndim != 2:
                print(f"Skipping {key}, Unable to visualize {fisher_cpu.ndim}D")
                continue

            # plot
            plt.figure(figsize=(6, 4))
            im = plt.imshow(fisher_cpu, cmap="viridis", aspect="auto")
            plt.colorbar(im)
            plt.title(f"Thresholded Fisher: {key}")
            plt.tight_layout()

            if save_fig:
                os.makedirs(save_dir, exist_ok=True)
                file_name = os.path.join(
                    save_dir, f"thres_fisher_{i}_{key.replace('.', '_')}.png"
                )
                plt.savefig(file_name)
                plt.close()
                print(f"- Saved to: {file_name}")
            else:
                plt.show()
                plt.close()

    def compare_all_fishers(
        self,
        file_path1,
        file_path2,
        task_id1=1,
        task_id2=2,
        save_dir="models/ewc_diff/figs",
        save_fig=False,
    ):
        print(
            f"\n[### Comparing All Fisher Matrices from task {task_id1} and {task_id2} ###]"
        )

        # 加载两个任务的 Fisher 字典
        fisher1 = self._load_single_fisher(file_path1, task_id1)
        fisher2 = self._load_single_fisher(file_path2, task_id2)

        os.makedirs(save_dir, exist_ok=True)

        for key in fisher1.keys():
            if key not in fisher2:
                print(f"[Warning] Key {key} not in second fisher dict, skipping...")
                continue

            # 特殊处理 koopman 中的 K_state_fisher
            if key == "_koopman_propagator.off_diagonal":
                continue
            elif key == "_koopman_propagator.diagonal":
                K1 = self._build_K_fisher(fisher1)
                K2 = self._build_K_fisher(fisher2)
                if K1 is None or K2 is None:
                    print("[Warning] Missing Koopman matrix components, skipping...")
                    continue
                diff_tensor = K1 - K2
            else:
                tensor1 = fisher1[key]
                tensor2 = fisher2[key]
                if tensor1.shape != tensor2.shape:
                    print(f"[Warning] Shape mismatch for key {key}, skipping...")
                    continue
                diff_tensor = tensor1 - tensor2

            diff_cpu = diff_tensor.detach().cpu().numpy()

            if diff_cpu.ndim == 1:
                diff_cpu = diff_cpu.reshape(1, -1)
            elif diff_cpu.ndim != 2:
                print(f"[Skipping] {key}, shape not suitable for visualization.")
                continue

            plt.figure(figsize=(6, 4))
            im = plt.imshow(diff_cpu, cmap="bwr", aspect="auto")
            plt.colorbar(im)
            plt.title(f"Fisher Diff: {key}")
            plt.tight_layout()

            if save_fig:
                file_name = os.path.join(save_dir, f"diff_{key.replace('.', '_')}.png")
                plt.savefig(file_name)
                plt.close()
                print(f"- Saved diff figure: {file_name}")
            else:
                plt.show()
                plt.close()

    def get_high_fisher_ratio(self, threshold, include_matrix=False):
        """
        计算所有Fisher值中，大于threshold的比例（以百分比形式返回）
        """
        total_elements = 0
        above_threshold = 0

        for key, tensor in self.fisher_dict.items():
            if key == "_koopman_propagator.off_diagonal":
                continue
            elif key == "_koopman_propagator.diagonal":
                if not include_matrix:
                    continue
                self.preprocess_K()
                tensor = self.K_state_fisher

            count = (tensor > threshold).sum().item()
            total = tensor.numel()
            above_threshold += count
            total_elements += total

        if total_elements == 0:
            print("[Warning] No valid elements in Fisher matrix to compute ratio.")
            return 0.0

        ratio = (above_threshold / total_elements) * 100
        print(
            f"\n[INFO] Percentage of Fisher values above threshold {threshold}: {ratio:.2f}%\n"
        )

    def get_layerwise_high_fisher_ratio(self, threshold):
        """
        输出每层参数中，Fisher值大于阈值的占比（百分比）
        """
        print(f"[INFO] Layer-wise Fisher value ratio above threshold {threshold}:")
        for key, tensor in self.fisher_dict.items():
            if key == "_koopman_propagator.off_diagonal":
                continue
            elif key == "_koopman_propagator.diagonal":
                self.preprocess_K()
                tensor = self.K_state_fisher

            total = tensor.numel()
            above = (tensor > threshold).sum().item()
            ratio = (above / total) * 100 if total > 0 else 0.0
            print(f"  - {key:<40}: {ratio:.2f}% ({above}/{total})")

    def _load_single_fisher(self, file_path, task_id):
        ckpt = torch.load(file_path)
        fisher_dict = ckpt.get("fisher_dict", {})
        if isinstance(fisher_dict, dict) and isinstance(
            list(fisher_dict.values())[0], dict
        ):
            return list(fisher_dict.values())[task_id - 1]
        return fisher_dict

    def _build_K_fisher(self, fisher_dict):
        off_diag = fisher_dict.get("_koopman_propagator.off_diagonal", None)
        diag = fisher_dict.get("_koopman_propagator.diagonal", None)
        if off_diag is None or diag is None:
            return None
        return torch.diag(diag) + off_diag - off_diag.t()
