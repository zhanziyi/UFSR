# -*- coding: utf-8 -*-

"""
@author: Ziyi Zhan
@unit: College of Environment and Ecology, Chongqing University, Chongqing 400045, China
@email: zhan20010727@163.com
@date: July 1st, 2025
@version: 1.0.0
@description: This code is an example of calling the UFSR model.
@usage: By editing dictionary parameters, modify UFSR model parameters and training parameters.
"""




import UFSR.NetModel
import UFSR.Optimizer
import UFSR.Loss



ufsr_all_net_dir = {
    "Baseline": {
      "name": "Baseline",
      "parameter": {
        "in_channels": 1,
        "out_channels": 1,
        "channels": 32,
        "num_rcb": 8,
        "upscale": 1,
        "enable_clamp": True,
        "clamp_min": 0,
        "clamp_max": 1
      }
    },
    "UFSR": {
      "name": "UFSR",
      "parameter": {
        "in_channels": 6,
        "out_channels": 1,
        "channels": 64,
        "num_rcb": 16,
        "upscale": 1,
        "enable_clamp": True,
        "clamp_min": 0,
        "clamp_max": 1,
        "cbam_ratio": 4,
        "cbam_kernel": 5
      }
    },
    "UFSR-SP": {
      "name": "UFSR-SP",
      "parameter": {
        "in_channels": 6,
        "out_channels": 1,
        "channels": 64,
        "num_rcb": 16,
        "upscale": 1,
        "enable_clamp": True,
        "clamp_min": 0,
        "clamp_max": 1,
        "cbam_ratio": 4
      }
    },
    "UFSR-CH": {
      "name": "UFSR-CH",
      "parameter": {
        "in_channels": 6,
        "out_channels": 1,
        "channels": 64,
        "num_rcb": 16,
        "upscale": 1,
        "enable_clamp": True,
        "clamp_min": 0,
        "clamp_max": 1,
        "cbam_kernel": 5
      }
    }
  }



ufsr_all_optimizer_dir = {
    "Adam": {
      "name": "Adam",
      "parameter": {
        "lr": 0.0001,
        "betas": [
          0.9,
          0.999
        ],
        "eps": 1e-08,
        "weight_decay": 0
      }
    }
  }


sr_all_loss_dir = {
    "MSE": {
      "name": "MSE",
      "parameter": {}
    },
    "WeightMSE": {
      "name": "WeightMSE",
      "parameter": {
        "threshold": 0.01,
        "weight_value": 6
      }
    },
    "MAPELoss": {
      "name": "MAPELoss",
      "parameter": {}
    },
    "Perception": {
      "name": "SSIM",
      "parameter":  {
        "window_size": 6,
        "sigma": 1.5,
        "k1": 0.01,
        "k2": 0.03,
        "max_pixel_value": 1
      }
    },
    "Statistics": {
      "name": "Statistics",
      "parameter":  {
        "threshold": 0.01
      }
    },
    "Grid": {
      "name": "Grid",
      "parameter":  {
        "threshold": 0.01,
        "beta1": 1,
        "beta2": 10,
        "beta3": 1
      }
    },
    "SPSC": {
      "name": "SPSC",
      "parameter":  {
        "threshold": 0.01,
        "alpha": 1,
        "beta": 1,
        "gamma": 1,
        "beta1": 1,
        "beta2": 10,
        "beta3": 1,
        "window_size": 6,
        "sigma": 1.5,
        "k1": 0.01,
        "k2": 0.03,
        "max_pixel_value": 1
      }
    }
  }



ufsr_device = "cuda"

ufsr_net_dir = ufsr_all_net_dir["UFSR"]

ufsr_optimizer_dir = ufsr_all_optimizer_dir["Adam"]

ufsr_loss_dir = sr_all_loss_dir["SPSC"]

ufsr_net = UFSR.NetModel.get_net_model_1(ufsr_net_dir).to(ufsr_device)

ufsr_optimizer = UFSR.Optimizer.get_optimizer_1(ufsr_net.parameters(), ufsr_optimizer_dir)

ufsr_loss_function = UFSR.Loss.get_loss_1(ufsr_loss_dir).to(ufsr_device)


