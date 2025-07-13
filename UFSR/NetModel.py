import math
import torch
from torch import nn, Tensor
import UFSR.Error


class _channel_attention_module(nn.Module):
    def __init__(self, ch, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(ch, ch // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // ratio, ch, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x1 = self.mlp(x1)
        x2 = self.max_pool(x).squeeze(-1).squeeze(-1)
        x2 = self.mlp(x2)
        feats = x1 + x2
        feats = self.sigmoid(feats).unsqueeze(-1).unsqueeze(-1)
        refined_feats = x * feats
        return refined_feats

class _spatial_attention_module(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)
        feats = torch.cat([x1, x2], dim=1)
        feats = self.conv(feats)
        feats = self.sigmoid(feats)
        refined_feats = x * feats
        return refined_feats

class _only_spatial_attention_module(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)
        feats = torch.cat([x1, x2], dim=1)
        feats = self.conv(feats)
        feats = self.sigmoid(feats)
        refined_feats = x * feats
        return refined_feats

class _CBAM(nn.Module):
    def __init__(self, channels, ratio, kernel_size):
        super().__init__()
        self.ca = _channel_attention_module(channels, ratio)
        self.sa = _spatial_attention_module(kernel_size)
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class _CBAMC(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()
        self.ca = _channel_attention_module(channels, ratio)
    def forward(self, x):
        x = self.ca(x)
        return x

class _CBAMS(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.sa = _only_spatial_attention_module(kernel_size)
    def forward(self, x):
        x = self.sa(x)
        return x


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.rcb(x)
        x = torch.add(x, identity)
        return x


class _CbamResidualConvBlock(nn.Module):
    def __init__(self, channels: int, cbam_ratio: int, cbam_kernel: int) -> None:
        super(_CbamResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            _CBAM(channels, cbam_ratio, cbam_kernel)
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.rcb(x)
        x = torch.add(x, identity)
        return x

class _CbamChResidualConvBlock(nn.Module):
    def __init__(self, channels: int, cbam_ratio: int) -> None:
        super(_CbamChResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            _CBAMC(channels, cbam_ratio)
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.rcb(x)
        x = torch.add(x, identity)
        return x

class _CbamSpResidualConvBlock(nn.Module):
    def __init__(self, channels: int, cbam_kernel: int) -> None:
        super(_CbamSpResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            _CBAMS(cbam_kernel)
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.rcb(x)
        x = torch.add(x, identity)
        return x


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample_block(x)

        return x

class Test1(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 16,
            num_rcb: int = 16,
            upscale: int = 2,
    ) -> None:
        super(Test1, self).__init__()
        # Low frequency information extraction layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # High frequency information extraction block
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # High-frequency information linear fusion layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        # zoom block
        upsampling = []
        if upscale == 2 or upscale == 4 or upscale == 8:
            for _ in range(int(math.log(upscale, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        else:
            raise NotImplementedError(f"Upscale factor `{upscale}` is not support.")
        self.upsampling = nn.Sequential(*upsampling)

        # reconstruction block
        # self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (2, 2), (4, 4))

        # Initialize neural network weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)
        return x


class Baseline(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
            num_rcb: int = 16,
            upscale: int = 1,
            enable_clamp: bool = False,
            clamp_min:float = 0,
            clamp_max:float = 1
    ) -> None:
        super(Baseline, self).__init__()
        self.enable_clamp = enable_clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.upscale = upscale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )
        upsampling = []
        if upscale == 1:
            pass #NOSONAR
        elif upscale == 2 or upscale == 4 or upscale == 8:
            for _ in range(int(math.log(upscale, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        else:
            UFSR.Error.error_exit_1(f"Upscale factor `{upscale}` is not support.")
        self.upsampling = nn.Sequential(*upsampling)
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def forward(self, x: Tensor) -> Tensor:
        if self.enable_clamp and self.upscale != 1:
            return self._forward_upscale_clamp(x)
        elif self.enable_clamp and self.upscale == 1:
            return self._forward_clamp(x)
        elif not self.enable_clamp and self.upscale != 1:
            return self._forward_upscale(x)
        elif not self.enable_clamp and self.upscale == 1:
            return self._forward(x)
        else:
            UFSR.Error.error_exit_1("Forward Error ")

    def _forward_upscale_clamp(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)
        x = torch.clamp_(x, 0.0, 1.0)
        return x
    
    def _forward_upscale(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)
        return x
    
    def _forward_clamp(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.conv3(x)
        x = torch.clamp_(x, 0.0, 1.0)
        return x
    
    def _forward(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.conv3(x)
        return x




class UFSR1(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
            num_rcb: int = 16,
            upscale: int = 1,
            enable_clamp: bool = False,
            clamp_min: float = 0,
            clamp_max: float = 1,
            cbam_ratio: int = 8,
            cbam_kernel: int = 7
    ) -> None:
        super(UFSR1, self).__init__()
        self.enable_clamp = enable_clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.upscale = upscale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU()
        )
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_CbamResidualConvBlock(channels, cbam_ratio, cbam_kernel))
        self.trunk = nn.Sequential(*trunk)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )
        upsampling = []
        if upscale == 1:
            pass #NOSONAR
        elif upscale == 2 or upscale == 4 or upscale == 8:
            for _ in range(int(math.log(upscale, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        else:
            UFSR.Error.error_exit_1(f"Upscale factor `{upscale}` is not support.")
        self.upsampling = nn.Sequential(*upsampling)
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def forward(self, x: Tensor) -> Tensor:
        if self.enable_clamp and self.upscale != 1:
            return self._forward_upscale_clamp(x)
        elif self.enable_clamp and self.upscale == 1:
            return self._forward_clamp(x)
        elif not self.enable_clamp and self.upscale != 1:
            return self._forward_upscale(x)
        elif not self.enable_clamp and self.upscale == 1:
            return self._forward(x)
        else:
            UFSR.Error.error_exit_1("Forward Error  ")

    def _forward_upscale_clamp(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)
        x = torch.clamp_(x, 0.0, 1.0)
        return x
    
    def _forward_upscale(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)
        return x
    
    def _forward_clamp(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.conv3(x)
        x = torch.clamp_(x, 0.0, 1.0)
        return x
    
    def _forward(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.conv3(x)
        return x



class UFSR_SP(nn.Module): #NOSONAR
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
            num_rcb: int = 16,
            upscale: int = 1,
            enable_clamp: bool = False,
            clamp_min: float = 0,
            clamp_max: float = 1,
            cbam_ratio: int = 8
    ) -> None:
        super(UFSR_SP, self).__init__()
        self.enable_clamp = enable_clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.upscale = upscale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU()
        )
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_CbamChResidualConvBlock(channels, cbam_ratio))
        self.trunk = nn.Sequential(*trunk)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )
        upsampling = []
        if upscale == 1:
            pass #NOSONAR
        elif upscale == 2 or upscale == 4 or upscale == 8:
            for _ in range(int(math.log(upscale, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        else:
            UFSR.Error.error_exit_1(f"Upscale factor `{upscale}` is not support.")
        self.upsampling = nn.Sequential(*upsampling)
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def forward(self, x: Tensor) -> Tensor:
        if self.enable_clamp and self.upscale != 1:
            return self._forward_upscale_clamp(x)
        elif self.enable_clamp and self.upscale == 1:
            return self._forward_clamp(x)
        elif not self.enable_clamp and self.upscale != 1:
            return self._forward_upscale(x)
        elif not self.enable_clamp and self.upscale == 1:
            return self._forward(x)
        else:
            UFSR.Error.error_exit_1("Forward Error")

    def _forward_upscale_clamp(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)
        x = torch.clamp_(x, 0.0, 1.0)
        return x
    
    def _forward_upscale(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)
        return x
    
    def _forward_clamp(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.conv3(x)
        x = torch.clamp_(x, 0.0, 1.0)
        return x
    
    def _forward(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.conv3(x)
        return x


class UFSR_CH(nn.Module): #NOSONAR
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
            num_rcb: int = 16,
            upscale: int = 1,
            enable_clamp: bool = False,
            clamp_min: float = 0,
            clamp_max: float = 1,
            cbam_kernel: int = 7
    ) -> None:
        super(UFSR_CH, self).__init__()
        self.enable_clamp = enable_clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.upscale = upscale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU()
        )
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_CbamSpResidualConvBlock(channels, cbam_kernel))
        self.trunk = nn.Sequential(*trunk)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )
        upsampling = []
        if upscale == 1:
            pass #NOSONAR
        elif upscale == 2 or upscale == 4 or upscale == 8:
            for _ in range(int(math.log(upscale, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        else:
            UFSR.Error.error_exit_1(f"Upscale factor `{upscale}` is not support.")
        self.upsampling = nn.Sequential(*upsampling)
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def forward(self, x: Tensor) -> Tensor:
        if self.enable_clamp and self.upscale != 1:
            return self._forward_upscale_clamp(x)
        elif self.enable_clamp and self.upscale == 1:
            return self._forward_clamp(x)
        elif not self.enable_clamp and self.upscale != 1:
            return self._forward_upscale(x)
        elif not self.enable_clamp and self.upscale == 1:
            return self._forward(x)
        else:
            UFSR.Error.error_exit_1("Forward Error")

    def _forward_upscale_clamp(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)
        x = torch.clamp_(x, 0.0, 1.0)
        return x
    
    def _forward_upscale(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)
        return x
    
    def _forward_clamp(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.conv3(x)
        x = torch.clamp_(x, 0.0, 1.0)
        return x
    
    def _forward(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.conv3(x)
        return x


def get_net_model_1(model_dir: dict[str,int|bool|float|str|dict|None]) -> nn.Module:
    all_model = {
        "Baseline": Baseline,
        "UFSR": UFSR1,
        "UFSR-SP": UFSR_SP,
        "UFSR-CH": UFSR_CH
        }
    if model_dir["name"] in all_model:
        return all_model[model_dir["name"]](**model_dir["parameter"])
    else:
        UFSR.Error.error_exit_1(f"Model Name: {model_dir["name"]} Does Not Exist")
    


