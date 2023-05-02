from typing import Tuple

import numpy as np
import torch
from torch import nn

from . import game


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 'same')
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(channels, channels * 2)

    def forward(self, x: torch.Tensor):
        shortcut = x
        n, c, h, w = x.size()
        x = self.pool(x)
        x = torch.reshape(x, (n, c))
        x = self.linear(x)
        x = torch.reshape(x, (n, c * 2, 1, 1))
        shift, scale = torch.split(x, c, dim=1)
        return shortcut * scale + shift


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 'same')
        self.norm2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels)

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.se(x)
        return shortcut + x


class Network(nn.Module):
    def __init__(self, channels: int, blocks: int):
        super().__init__()
        assert game.Game.NUM_ACTIONS == game.Game.STATE_WIDTH
        self.init_conv = ConvLayer(game.Game.STATE_LAYERS, channels)
        self.common = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(
            ConvLayer(channels, channels),
            nn.Conv2d(channels, 1, (game.Game.BOARD_HEIGHT, 1), 1, 'valid'),
            nn.Flatten(),
            nn.Softmax(dim=-1),
        )
        self.wdl = nn.Sequential(
            ConvLayer(channels, 8),
            nn.Flatten(),
            nn.Linear(game.Game.STATE_HEIGHT * game.Game.STATE_WIDTH * 8, 128),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor):
        x = self.init_conv(x)
        x = self.common(x)
        policy = self.policy(x)
        wdl = self.wdl(x)
        return policy, wdl


class Model:
    def __init__(self, channels: int, blocks: int, learning_rate: float, device: torch.device = None):
        self.criterion = nn.MSELoss()
        if device is None:
            device = 'cpu'
        self.device = device
        self.net = Network(channels, blocks).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), learning_rate)

    @torch.no_grad()
    def policy_function(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        state = state[np.newaxis, ...].astype(np.float32)
        state = torch.from_numpy(state)
        act, val = self.net(state)
        act = act.detach().cpu().numpy()
        val = val.detach().cpu().numpy()
        return act[0], val[0]

    def load(self, file_name: str):
        state_dict = torch.load(file_name, map_location=self.device)
        self.net.load_state_dict(state_dict)

    def save(self, file_name: str):
        torch.save(self.net.state_dict(), file_name)

    def train_step(self, states: torch.Tensor, y_pol: torch.Tensor, y_wdl: torch.Tensor):
        states = states.to(self.device)
        y_pol = y_pol.to(self.device)
        y_wdl = y_wdl.to(self.device)
        pred_pol, pred_wdl = self.net(states)
        pol_loss = self.criterion(pred_pol, y_pol)
        wdl_loss = self.criterion(pred_wdl, y_wdl)
        total_loss = pol_loss + wdl_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def train(self, states: np.ndarray, y_pol: np.ndarray, y_wdl: np.ndarray, batch_size: int):
        states = states.astype(np.float32)
        y_pol = y_pol.astype(np.float32)
        y_wdl = y_wdl.astype(np.float32)
        dataset = torch.utils.data.TensorDataset(
            *map(torch.from_numpy, (states, y_pol, y_wdl)))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)
        for states, y_pol, y_wdl in loader:
            self.train_step(states, y_pol, y_wdl)
