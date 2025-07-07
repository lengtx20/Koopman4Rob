""" This file provides the implementation of the training procedure """

import os
import numpy as np
import torch
import torch.optim as optim
from torch.nn import MSELoss
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils import smooth_curve

class KoopmanRunner:
    def __init__(self, model, ewc_model, train_data, val_data, optimizer, loss_fn, device, normalize=False, ewc_lambda=0.0):
        """
            model:      Deep Koopman model
            normalize:  Current normalization relies on pre-defined mean and std value.
                        This is to assure the consistency of the model.
        """
        self.model = model
        self.ewc_model = ewc_model
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.normalize = normalize
        self.ewc_lambda = ewc_lambda

        if self.normalize:
            self.state_mean = np.array([-1.27054915,  0.94617132, -0.32996104,  5.84603260])
            self.state_std  = np.array([ 5.24317368,  4.13141372,  2.37722976,  2.74760289])
            self.action_mean = np.array([-1.14226600, -0.00369027])
            self.action_std  = np.array([ 1.66356851,  0.32236316])
        else:
            self.mean = None
            self.std = None
    
    def _process_batch(self, data, i):
        '''
            Split data into x_t, a_t, x_t1. Single sample at a time.
            If self.normalize = True, then sample retured will be normalized.
        '''
        sample = data[i]
        x_t = sample[:self.model.state_dim]
        a_t = sample[self.model.state_dim:self.model.state_dim + self.model.action_dim]
        x_t1 = sample[-self.model.state_dim:]
        if self.normalize:
            x_t = (x_t - self.state_mean) / self.state_std
            a_t = (a_t - self.action_mean) / self.action_std
            x_t1 = (x_t1 - self.state_mean) / self.state_std
        x_t = torch.tensor(x_t, dtype=torch.float32).to(self.device).unsqueeze(0)
        a_t = torch.tensor(a_t, dtype=torch.float32).to(self.device).unsqueeze(0)
        x_t1 = torch.tensor(x_t1, dtype=torch.float32).to(self.device).unsqueeze(0)
        return x_t, a_t, x_t1

    def _denormalize(self, data):
        if not self.normalize:
            return data
        state_dim = self.model.state_dim
        action_dim = self.model.action_dim

        x_t = data[:state_dim] * self.state_std + self.state_mean
        a_t = data[state_dim:state_dim + action_dim] * self.action_std + self.action_mean
        x_t1 = data[state_dim + action_dim:state_dim * 2 + action_dim] * self.state_std + self.state_mean
        pred_x_t1 = data[-state_dim:] * self.state_std + self.state_mean

        return np.concatenate((x_t, a_t, x_t1, pred_x_t1))

    def _evaluate_loss(self, data):
        """
            Compute the prediction loss by: loss = loss_fn (pred_x_t1, x_t1)
        """
        if data is None:
            return None
        
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i in range(data.shape[0]):
                x_t, a_t, x_t1 = self._process_batch(data, i)
                pred_x_t1 = self.model(x_t, a_t, False)
                loss = self.loss_fn(pred_x_t1, x_t1)
                total_loss += loss.item()
        return total_loss / data.shape[0]

    def load_fisher(self, fisher_path, task_id=1):
        self.ckpt = torch.load(fisher_path)
        self.fisher_dict = self.ckpt.get('fisher_dict', {})
        print("[INFO] fisher_dict length:", len(self.fisher_dict))
        print("[INFO] fisher_dict keys:", self.fisher_dict.keys())
        if isinstance(self.fisher_dict, dict) and isinstance(list(self.fisher_dict.values())[0], dict):
            self.fisher_dict = list(self.fisher_dict.values())[task_id-1]

    def register_gradient_masks(self, threshold_mode, ewc_threshold):
        def create_mask(fisher_tensor):
            mask = torch.ones_like(fisher_tensor)
            if threshold_mode == 'value':
                mask[fisher_tensor < ewc_threshold] = 0
            elif threshold_mode == 'neural_ratio':
                thresh_val = torch.quantile(fisher_tensor.view(-1), ewc_threshold)
                mask[fisher_tensor < thresh_val] = 0
            elif threshold_mode == 'weight_ratio':
                min_val = fisher_tensor.min()
                max_val = fisher_tensor.max()
                thresh_val = min_val + ewc_threshold * (max_val - min_val)
                mask[fisher_tensor < thresh_val] = 0
            else:
                raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")
            return mask.to(self.device)

        # -------- param A --------
        param_A = self.model.A
        fisher_A = self.fisher_dict.get('A', None)
        if fisher_A is not None and fisher_A.shape == param_A.shape:
            mask_A = create_mask(fisher_A)
            param_A.register_hook(lambda grad: grad * mask_A)
        else:
            print("[Warning] No valid Fisher info for A")

        # -------- param B --------
        param_B = self.model.B
        fisher_B = self.fisher_dict.get('B', None)
        if fisher_B is not None and fisher_B.shape == param_B.shape:
            mask_B = create_mask(fisher_B)
            param_B.register_hook(lambda grad: grad * mask_B)
        else:
            print("[Warning] No valid Fisher info for B")

        # -------- encoder.layers.0.weight --------
        param_0_w = self.model.encoder.layers[0].weight
        fisher_0_w = self.fisher_dict.get('encoder.layers.0.weight', None)
        if fisher_0_w is not None and fisher_0_w.shape == param_0_w.shape:
            mask_0_w = create_mask(fisher_0_w)
            param_0_w.register_hook(lambda grad: grad * mask_0_w)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.0.weight")

        # -------- encoder.layers.0.bias --------
        param_0_b = self.model.encoder.layers[0].bias
        fisher_0_b = self.fisher_dict.get('encoder.layers.0.bias', None)
        if fisher_0_b is not None and fisher_0_b.shape == param_0_b.shape:
            mask_0_b = create_mask(fisher_0_b)
            param_0_b.register_hook(lambda grad: grad * mask_0_b)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.0.bias")

        # -------- encoder.layers.2.weight --------
        param_2_w = self.model.encoder.layers[2].weight
        fisher_2_w = self.fisher_dict.get('encoder.layers.2.weight', None)
        if fisher_2_w is not None and fisher_2_w.shape == param_2_w.shape:
            mask_2_w = create_mask(fisher_2_w)
            param_2_w.register_hook(lambda grad: grad * mask_2_w)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.2.weight")

        # -------- encoder.layers.2.bias --------
        param_2_b = self.model.encoder.layers[2].bias
        fisher_2_b = self.fisher_dict.get('encoder.layers.2.bias', None)
        if fisher_2_b is not None and fisher_2_b.shape == param_2_b.shape:
            mask_2_b = create_mask(fisher_2_b)
            param_2_b.register_hook(lambda grad: grad * mask_2_b)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.2.bias")

        # -------- encoder.layers.4.weight --------
        param_4_w = self.model.encoder.layers[4].weight
        fisher_4_w = self.fisher_dict.get('encoder.layers.4.weight', None)
        if fisher_4_w is not None and fisher_4_w.shape == param_4_w.shape:
            mask_4_w = create_mask(fisher_4_w)
            param_4_w.register_hook(lambda grad: grad * mask_4_w)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.4.weight")

        # -------- encoder.layers.4.bias --------
        param_4_b = self.model.encoder.layers[4].bias
        fisher_4_b = self.fisher_dict.get('encoder.layers.4.bias', None)
        if fisher_4_b is not None and fisher_4_b.shape == param_4_b.shape:
            mask_4_b = create_mask(fisher_4_b)
            param_4_b.register_hook(lambda grad: grad * mask_4_b)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.4.bias")

        # -------- encoder.layers.6.weight --------
        param_6_w = self.model.encoder.layers[6].weight
        fisher_6_w = self.fisher_dict.get('encoder.layers.6.weight', None)
        if fisher_6_w is not None and fisher_6_w.shape == param_6_w.shape:
            mask_6_w = create_mask(fisher_6_w)
            param_6_w.register_hook(lambda grad: grad * mask_6_w)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.6.weight")

        # -------- encoder.layers.6.bias --------
        param_6_b = self.model.encoder.layers[6].bias
        fisher_6_b = self.fisher_dict.get('encoder.layers.6.bias', None)
        if fisher_6_b is not None and fisher_6_b.shape == param_6_b.shape:
            mask_6_b = create_mask(fisher_6_b)
            param_6_b.register_hook(lambda grad: grad * mask_6_b)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.6.bias")

    def train(self, max_epochs=100, save_model=True, model_dir=None, task_id=1, ewc_regularization=False, fisher_path=None, threshold_mode=None, ewc_threshold=0.0):
        """
            Called by 'train' mode.
        """
        self.model.train()
        best_val_loss = float('inf')

        if threshold_mode is not None:
            self.fisher_dict = None
            self.load_fisher(fisher_path=fisher_path)

        # training step
        epoch_bar = tqdm(range(max_epochs), desc="[Training]", position=0)
        for epoch in epoch_bar:
            total_loss = 0

            pbar = tqdm(range(self.train_data.shape[0]), desc=f"[Train] Epoch {epoch+1}/{max_epochs}", leave=False)
            self.register_gradient_masks(threshold_mode=threshold_mode, ewc_threshold=ewc_threshold)
            
            for i in pbar:
            # for i in range(self.train_data.shape[0]):
                x_t, a_t, x_t1 = self._process_batch(self.train_data, i)

                self.optimizer.zero_grad()
                pred_x_t1 = self.model(x_t, a_t, False)
                loss = self.loss_fn(pred_x_t1, x_t1)
                if ewc_regularization:
                    if self.ewc_model is not None and self.ewc_model.fisher is not None:
                        loss += self.ewc_lambda * self.ewc_model.penalty(self.model)
                # register mask to disable partial params
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / self.train_data.shape[0]
            val_loss = self._evaluate_loss(self.val_data)

            if val_loss is not None:
                epoch_bar.set_postfix({
                    'Epoch': epoch + 1,
                    'TrainLoss': f"{train_loss:.4f}",
                    'ValLoss': f"{val_loss:.4f}"
                })
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"[Epoch {epoch+1}] Validation loss improved to {val_loss:.4f}.")
            else:
                epoch_bar.set_postfix({'TrainLoss': f"{train_loss:.4f}"})
                print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} (no validation)")

        # save model
        if save_model and model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
            self.model.save(model_dir=model_dir)
            print(f"[Runner] Model saved to {model_dir}")
        else:
            print("[INFO] No Koopman model saved")

        if save_model and model_dir is not None:
            self.model.save(model_dir=model_dir)
            print(f"[Runner] Model saved to {model_dir}")
        else:
            print("[INFO] No Koopman model saved")

        # compute fisher after training and save fisher info
        print("[INFO] Computing Fisher Information after training...")
        if self.ewc_model is not None:
            train_tensor = torch.tensor(self.train_data, dtype=torch.float32).to(self.device)
            fisher = self.ewc_model.compute_fisher(train_tensor, batch_size=64)
            self.ewc_model.fisher = fisher
            print("[INFO] Fisher matrix updated.")
            if save_model and model_dir is not None:
                self.ewc_model.save(model_dir=model_dir, task_id=task_id)
        else:
            print("[INFO] No EWC model attached or invalid.")

    def test(self, model_dir=None, show_plot=True):
        """
            Called by 'test' mode.
            If show_plot is true, some visualization of the data trajectory will be shown.
        """
        self.model.load(model_dir=model_dir)
        self.model.eval()
        self.traj = []
        total_loss = 0
        with torch.no_grad():
            for i in range(self.train_data.shape[0]):
                x_t, a_t, x_t1 = self._process_batch(self.train_data, i)
                pred_x_t1 = self.model(x_t, a_t, False)
                loss = self.loss_fn(pred_x_t1, x_t1)
                total_loss += loss.item()

                x_t_np = x_t.squeeze(0).cpu().numpy()
                a_t_np = a_t.squeeze(0).cpu().numpy()
                x_t1_np = x_t1.squeeze(0).cpu().numpy()
                pred_x_t1_np = pred_x_t1.squeeze(0).cpu().numpy()

                if self.normalize:
                    combined = np.concatenate((x_t_np, a_t_np, x_t1_np, pred_x_t1_np), axis=-1)
                    denorm = self._denormalize(combined)
                    x_t_np = denorm[:self.model.state_dim]
                    a_t_np = denorm[self.model.state_dim:self.model.state_dim + self.model.action_dim]
                    x_t1_np = denorm[self.model.state_dim + self.model.action_dim:self.model.state_dim * 2 + self.model.action_dim]
                    pred_x_t1_np = denorm[-self.model.state_dim:]

                traj_item = np.concatenate((x_t_np, a_t_np, x_t1_np, pred_x_t1_np), axis=-1)
                self.traj.append(traj_item)

        self.average_loss = total_loss / self.train_data.shape[0]
        self.traj_np = np.array(self.traj)
        print(f"[Validate] Avg Loss: {self.average_loss:.4f}")
        if show_plot:
            self.plot_trajectory()

    def plot_trajectory(self, use_smooth=True):
        """
            If use_smooth, traj is smoothed using function 'smooth_curve' by doing interpolation.
            This function needs to be modified by user.
        """
        x_t = self.traj_np[:, :self.model.state_dim]
        pred_x_t1 = self.traj_np[:, -self.model.state_dim:]

        assert self.model.state_dim >= 4, "state_dim 至少为 4，才能绘制两个二维平面轨迹"

        plt.figure(figsize=(10, 5))
        
        # Plot for dim 0 & 1
        plt.subplot(1, 2, 1)
        if use_smooth:
            x_s, y_s = smooth_curve(x_t[:, 0], x_t[:, 1])
            x_p, y_p = smooth_curve(pred_x_t1[:, 0], pred_x_t1[:, 1])
        else:
            x_s, y_s = x_t[:, 0], x_t[:, 1]
            x_p, y_p = pred_x_t1[:, 0], pred_x_t1[:, 1]
        plt.plot(x_s, y_s, label='True Traj', color='blue')
        plt.plot(x_p, y_p, label='Predicted Traj', color='orange')
        plt.scatter(x_t[:, 0], x_t[:, 1], color='blue', s=10, alpha=0.3)
        plt.scatter(pred_x_t1[:, 0], pred_x_t1[:, 1], color='orange', s=10, alpha=0.3)
        plt.title('State[0] vs State[1]')
        plt.xlabel('State 0')
        plt.ylabel('State 1')
        plt.legend()
        plt.axis("equal")

        # Plot for dim 2 & 3
        plt.subplot(1, 2, 2)
        if use_smooth:
            x_s, y_s = smooth_curve(x_t[:, 2], x_t[:, 3])
            x_p, y_p = smooth_curve(pred_x_t1[:, 2], pred_x_t1[:, 3])
        else:
            x_s, y_s = x_t[:, 2], x_t[:, 3]
            x_p, y_p = pred_x_t1[:, 2], pred_x_t1[:, 3]
        plt.plot(x_s, y_s, label='True Traj', color='blue')
        plt.plot(x_p, y_p, label='Predicted Traj', color='orange')
        plt.scatter(x_t[:, 2], x_t[:, 3], color='blue', s=10, alpha=0.3)
        plt.scatter(pred_x_t1[:, 2], pred_x_t1[:, 3], color='orange', s=10, alpha=0.3)
        plt.title('State[2] vs State[3]')
        plt.xlabel('State 2')
        plt.ylabel('State 3')
        plt.legend()
        plt.axis("equal")

        plt.tight_layout()
        plt.show()


