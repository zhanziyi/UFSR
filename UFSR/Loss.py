import torch.nn as nn
import UFSR.Error
import torch
import torch.nn.functional as F


class WeightMSE(nn.Module):
    def __init__(self, threshold: float, weight_value: float) -> None:
        super(WeightMSE, self).__init__()
        self.threshold = threshold
        self.weight_value = weight_value

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(label) 
        weights[input < self.threshold] = self.weight_value
        loss = weights * (input - label) ** 2
        return loss.mean()


class MAPELoss(nn.Module):
    def __init__(self) -> None:
        super(MAPELoss, self).__init__()

    def forward(self, predicted, target) -> torch.Tensor:
        mask = predicted != 0
        if not torch.any(mask):
            return 0.0
        return torch.mean(torch.abs((target[mask] - predicted[mask]) / predicted[mask]))



class Perception(nn.Module):
    def __init__(self, window_size: int, sigma: float, k1: float, k2: float, max_pixel_value: int) -> None:
        super(Perception, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2
        self.max_pixel_value = max_pixel_value

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> float:            
        half_size = self.window_size // 2
        x = torch.arange(-half_size, half_size + 1, device=input.device)
        y = torch.arange(-half_size, half_size + 1, device=input.device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        mu1 = F.conv2d(input, kernel, padding=half_size, groups=1)
        mu2 = F.conv2d(label, kernel, padding=half_size, groups=1)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(input ** 2, kernel, padding=half_size, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(label ** 2, kernel, padding=half_size, groups=1) - mu2_sq
        sigma12 = F.conv2d(input * label, kernel, padding=half_size, groups=1) - mu1_mu2
        c1 = (self.k1 * self.max_pixel_value) ** 2
        c2 = (self.k2 * self.max_pixel_value) ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1 - ssim_map.mean(dim=(1, 2, 3)).mean()



class Statistics(nn.Module):
    def __init__(self, threshold: float, epsilon: float = 1e-8) -> None:
        super(Statistics, self).__init__()
        self.threshold = threshold
        self.epsilon = epsilon

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted_prob = torch.sigmoid((predicted - self.threshold)*10000)
        target_prob = torch.sigmoid((target - self.threshold)*10000)
        count_predicted = torch.mean(predicted_prob)
        count_target = torch.mean(target_prob) + self.epsilon
        return torch.abs(1 - count_predicted / count_target)

class Grid(nn.Module):
    def __init__(self, threshold, beta1: float, beta2: float, beta3: float) -> None:
        super(Grid, self).__init__()
        self.threshold = threshold
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
    
    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        mse1 = torch.mean((label - input) ** 2)
        mask = label >= self.threshold
        if torch.count_nonzero(mask) == 0: #NOSONAR
            mse2 = 0.0
        else:
            mse2 = torch.mean((label[mask] - input[mask]) ** 2)
        mask = label < self.threshold
        if torch.count_nonzero(mask) == 0: #NOSONAR
            mse3 = 0.0
        else:
            mse3 = torch.mean((label[mask] - input[mask]) ** 2)
        return self.beta1 * mse1 + self.beta2 * mse2 + self.beta3 * mse3


class SPSC(nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float, 
                 threshold, beta1: float, beta2: float, beta3: float, 
                 window_size: int, sigma: float, k1: float, k2: float, max_pixel_value: int, 
                 epsilon: float = 1e-8) -> None:
        super(SPSC, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.window_size = window_size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2
        self.max_pixel_value = max_pixel_value
    
    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        predicted_prob = torch.sigmoid((input - self.threshold)*10000)
        target_prob = torch.sigmoid((label - self.threshold)*10000)
        count_predicted = torch.mean(predicted_prob)
        count_target = torch.mean(target_prob) + self.epsilon
        ploss = torch.abs(1 - count_predicted / count_target)
        mse1 = torch.mean((label - input) ** 2)
        mask = label >= self.threshold
        if torch.count_nonzero(mask) == 0: #NOSONAR
            mse2 = 0.0
        else:
            mse2 = torch.mean((label[mask] - input[mask]) ** 2)
        mask = label < self.threshold
        if torch.count_nonzero(mask) == 0: #NOSONAR
            mse3 = 0.0
        else:
            mse3 = torch.mean((label[mask] - input[mask]) ** 2)
        mmse = self.beta1 * mse1 + self.beta2 * mse2 + self.beta3 * mse3
        half_size = self.window_size // 2
        x = torch.arange(-half_size, half_size + 1, device=input.device)
        y = torch.arange(-half_size, half_size + 1, device=input.device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        mu1 = F.conv2d(input, kernel, padding=half_size, groups=1)
        mu2 = F.conv2d(label, kernel, padding=half_size, groups=1)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(input ** 2, kernel, padding=half_size, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(label ** 2, kernel, padding=half_size, groups=1) - mu2_sq
        sigma12 = F.conv2d(input * label, kernel, padding=half_size, groups=1) - mu1_mu2
        c1 = (self.k1 * self.max_pixel_value) ** 2
        c2 = (self.k2 * self.max_pixel_value) ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        ssim = 1 - ssim_map.mean(dim=(1, 2, 3)).mean()
        return self.alpha * ploss + self.beta * mmse + self.gamma * ssim



def get_loss_1(loss_dir: dict[str,int|bool|float|str|dict|None]) -> nn.MSELoss:
    all_optimizer = {
        "MSE": nn.MSELoss,
        "WeightMSE": WeightMSE,
        "MAPELoss": MAPELoss,
        "Perception":Perception, 
        "Statistics": Statistics, 
        "Grid": Grid,
        "SPSC": SPSC
        }
    if loss_dir["name"] in all_optimizer:
        return all_optimizer[loss_dir["name"]](**loss_dir["parameter"])
    else:
        UFSR.Error.error_exit_1(f"Loss Name: {loss_dir["name"]} Does Not Exist")
