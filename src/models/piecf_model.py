"""
PIECF Model Architecture (ConvLSTM Spatial-Temporal Encoder)

Key innovation: Uses ConvLSTM to jointly model spatial and temporal patterns
in the multi-channel satellite data. Unlike the MLP baseline that pools
away spatial information, ConvLSTM preserves the [H, W] grid structure
throughout the encoder, enabling the model to distinguish thermal stress
across different regions (e.g., South China Sea vs. Bohai Sea).

Input:  [B, T, H, W, C_dyn] — dynamic features over a 26-week window
Output: regression, classification, warning, ssta_pred — all preserve [H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATA_CONFIG, SCS_CONFIG, MODEL_CONFIG


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell — replaces linear gates with convolutional gates.

    ConvLSTM extends LSTM by replacing matrix multiplications with convolutions,
    allowing the model to capture spatial correlations (e.g., thermal stress
    spreading to neighboring grid cells) while modeling temporal dynamics.

    Input:  x [B, C_in, H, W], hidden [B, C_hidden, H, W], cell [B, C_hidden, H, W]
    Output: hidden [B, C_hidden, H, W], cell [B, C_hidden, H, W]
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Combined convolution for all 4 gates (i, f, o, g)
        # Output: [B, 4 * hidden_channels, H, W]
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        self.bn = nn.BatchNorm2d(4 * hidden_channels)

    def forward(self, x, hidden, cell):
        """
        Args:
            x: [B, C_in, H, W] — input at current time step
            hidden: [B, C_hidden, H, W] — previous hidden state
            cell: [B, C_hidden, H, W] — previous cell state
        Returns:
            hidden: [B, C_hidden, H, W] — new hidden state
            cell: [B, C_hidden, H, W] — new cell state
        """
        combined = torch.cat([x, hidden], dim=1)  # [B, C_in + C_hidden, H, W]
        combined = self.conv(combined)             # [B, 4*C_hidden, H, W]
        combined = self.bn(combined)

        # Split into 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.hidden_channels, dim=1)

        # Apply activations
        i = torch.sigmoid(cc_i)  # input gate
        f = torch.sigmoid(cc_f)  # forget gate
        o = torch.sigmoid(cc_o)  # output gate
        g = torch.tanh(cc_g)     # candidate cell

        # Update cell and hidden state
        cell = f * cell + i * g       # [B, C_hidden, H, W]
        hidden = o * torch.tanh(cell)  # [B, C_hidden, H, W]

        return hidden, cell


class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM — processes the full time sequence.

    Input:  [B, T, C_in, H, W]
    Output: [B, T, C_hidden, H, W] — hidden states at all time steps
    """

    def __init__(self, input_channels, hidden_channels, num_layers=2, kernel_size=3):
        super().__init__()
        self.num_layers = num_layers

        # Stack multiple ConvLSTM layers
        cells = []
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            cells.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, x):
        """
        Args:
            x: [B, T, C_in, H, W] — input sequence
        Returns:
            outputs: [B, T, C_hidden, H, W] — hidden states at all time steps
        """
        B, T, C, H, W = x.shape

        # Initialize hidden and cell states for each layer
        h = [torch.zeros(B, self.cells[0].hidden_channels, H, W, device=x.device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.cells[0].hidden_channels, H, W, device=x.device)
             for _ in range(self.num_layers)]

        outputs = []
        for t in range(T):
            x_t = x[:, t]  # [B, C_in, H, W]
            for layer_idx in range(self.num_layers):
                h[layer_idx], c[layer_idx] = self.cells[layer_idx](
                    x_t, h[layer_idx], c[layer_idx]
                )
                x_t = h[layer_idx]  # feed to next layer
            outputs.append(h[-1])  # output from last layer

        return torch.stack(outputs, dim=1)  # [B, T, C_hidden, H, W]


class ChannelNormalizer(nn.Module):
    """Per-channel z-score normalization with NaN handling.

    Normalizes each dynamic feature channel to ~N(0,1) using pre-computed
    dataset statistics. NaN values (missing data) are set to 0 after
    normalization (= mean in normalized space).

    Input:  x [B, T, H, W, C], nan_mask [C, T, H, W] (True=valid)
    Output: x_normalized [B, T, H, W, C]
    """
    def __init__(self, mean, std):
        super().__init__()
        # mean, std: list of length C
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 1, 1, 1, -1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 1, 1, 1, -1))

    def forward(self, x, nan_mask=None):
        # x: [B, T, H, W, C]
        x = (x - self.mean) / self.std
        if nan_mask is not None:
            # nan_mask: [B, C, T, H, W] (from DataLoader) or [C, T, H, W]
            if nan_mask.dim() == 4:
                nan_mask = nan_mask.unsqueeze(0)  # [1, C, T, H, W]
            # → [B, T, H, W, C] for broadcasting
            nan_mask_t = nan_mask.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
            x = torch.where(nan_mask_t, x, torch.zeros_like(x))
        else:
            x = torch.nan_to_num(x, nan=0.0)
        return x


class PIECFModel(nn.Module):
    """ConvLSTM-based PIECF model — preserves spatial structure throughout.

    Architecture:
        1. Input projection: [B, T, H, W, C_dyn] → [B, T, C_hidden, H, W]
        2. ConvLSTM encoder: [B, T, C_hidden, H, W] → [B, T, hidden, H, W]
        3. Temporal attention: weighted sum over T → [B, hidden, H, W]
        4. Spatial pooling + MLP: → predictions

    This design preserves spatial information (H, W) through the encoder,
    enabling the model to learn region-specific thermal stress patterns.
    """

    def __init__(self, dynamic_channels=None, grid_size=None):
        super().__init__()
        # 优先使用外部传入的通道数 (对齐 SCS 数据集); 否则 fallback DATA_CONFIG
        if dynamic_channels is not None:
            self.C_dyn = dynamic_channels
            default_cfg = SCS_CONFIG if dynamic_channels == SCS_CONFIG['dynamic_channels'] else DATA_CONFIG
        else:
            default_cfg = SCS_CONFIG
            self.C_dyn = SCS_CONFIG['dynamic_channels']
        self.T = default_cfg['history_weeks']
        self.H = grid_size[0] if grid_size else default_cfg['grid_lat']
        self.W = grid_size[1] if grid_size else default_cfg['grid_lon']
        self.C_stat = default_cfg['static_channels']

        # Model hyperparameters
        hidden_channels = MODEL_CONFIG['convlstm_hidden_dims']  # e.g., [32, 64]
        num_layers = len(hidden_channels)
        attention_dim = MODEL_CONFIG['attention_dim']
        attention_heads = MODEL_CONFIG['attention_heads']
        dropout = MODEL_CONFIG['dropout']

        # === 0. Channel normalization (z-score with NaN handling) ===
        from configs.config import SCS_CHANNEL_MEAN, SCS_CHANNEL_STD
        self.channel_norm = ChannelNormalizer(SCS_CHANNEL_MEAN, SCS_CHANNEL_STD)

        # === 1. Input projection: C_dyn → hidden_channels[0] ===
        self.input_proj = nn.Sequential(
            nn.Conv2d(self.C_dyn, hidden_channels[0], kernel_size=1),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(),
        )

        # === 2. Multi-layer ConvLSTM encoder ===
        self.convlstm_layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels[i - 1] if i > 0 else hidden_channels[0]
            out_ch = hidden_channels[i]
            self.convlstm_layers.append(
                ConvLSTM(in_ch, out_ch, num_layers=1, kernel_size=3)
            )
            # Project residual if dimensions change
            if in_ch != out_ch:
                self.convlstm_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1))
            else:
                self.convlstm_layers.append(nn.Identity())

        self.hidden_channels = hidden_channels[-1]

        # === 3. Temporal self-attention ===
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_channels,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_proj = nn.Linear(self.hidden_channels, attention_dim)

        # Learned temporal aggregation: weighted sum over T (replaces mean-pool)
        # Projects each timestep's hidden state to a scalar attention weight
        self.temporal_agg = nn.Linear(self.hidden_channels, 1)

        # === 4. Prediction heads (per-timestep, per-pixel) ===
        # 关键改进: 每个时间步独立预测 (不再从单一 context 向量展开到 T)
        # 消除信息瓶颈: hidden [B,T,C_hid,H,W] → 每个 timestep 独立过 head
        regression_hidden = MODEL_CONFIG['regression_hidden']
        warning_hidden = MODEL_CONFIG['warning_hidden']

        # 每个 head 输出 1 个值 (per timestep), 输入 [..., C_hid] → 输出 [..., 1]
        self.regression_head = nn.Sequential(
            nn.Linear(self.hidden_channels, regression_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(regression_hidden, 1),
            nn.Sigmoid()
        )

        # 分类头输出 logit (无 sigmoid)，配合 BCEWithLogitsLoss
        self.classification_head = nn.Sequential(
            nn.Linear(self.hidden_channels, regression_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(regression_hidden, 1)
            # 无 sigmoid — 在 loss 中用 BCEWithLogitsLoss
        )

        self.warning_head = nn.Sequential(
            nn.Linear(self.hidden_channels, warning_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(warning_hidden, 1),
            nn.Sigmoid()
        )

        # BAA 多级分类回归头: 预测连续 BAA 等级 [0, 3]
        self.baa_head = nn.Sequential(
            nn.Linear(self.hidden_channels, warning_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(warning_hidden, 1),
            nn.Sigmoid()  # → [0, 1], scaled ×3 in loss
        )

        # Hotspot 预测: 物理约束用 (Hotspot → DHW 热力学链路)
        # 与 ssta_head 同结构, 但目标是 CRW Hotspot (热应力指标, °C, ≥ 0)
        self.hotspot_head = nn.Sequential(
            nn.Linear(self.hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()  # Hotspot ≥ 0, 输出非负值
        )

        # Grad-CAM hooks
        self.gradients = None
        self.activations = None

    def register_hook(self):
        """Register forward/backward hooks for Grad-CAM explainability."""
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output[0].detach()
        target_layer = self.spatial_pool
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def forward_features(self, x_dynamic, x_static, land_mask, nan_mask=None):
        """Run shared backbone (norm + ConvLSTM + temporal attn) and return
        the per-timestep feature tensor that prediction heads consume.

        Returns dict:
            attn_out_reshaped: [B, H, W, T, C_hid] — features for head input
            hidden:           [B, T, C_hid, H, W]   — optional diagnostics
        """
        B, T, H, W, C = x_dynamic.shape

        # Ensure land_mask broadcasts correctly
        if land_mask is not None and land_mask.dim() == 2:
            land_mask = land_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # === 0. Channel normalization (z-score with NaN handling) ===
        x_dynamic = self.channel_norm(x_dynamic, nan_mask)

        # === 1. Input projection + permute for ConvLSTM ===
        # [B, T, H, W, C_dyn] → [B, T, C_dyn, H, W]
        x = x_dynamic.permute(0, 1, 4, 2, 3)  # [B, T, C_dyn, H, W]
        # Apply 1x1 conv per time step to project channels
        x = x.reshape(B * T, C, H, W)
        x = self.input_proj(x)  # [B*T, hidden_channels[0], H, W]
        x = x.reshape(B, T, -1, H, W)  # [B, T, hidden_0, H, W]

        # === 2. ConvLSTM encoder ===
        for i, (cell_layer, proj) in enumerate(
            zip(self.convlstm_layers[::2], self.convlstm_layers[1::2])
        ):
            residual = x  # [B, T, in_ch, H, W]
            x = cell_layer(x)  # [B, T, out_ch, H, W]
            # Project residual if needed and add
            res_ch = residual.shape[2]
            if res_ch != x.shape[2]:
                residual = residual.reshape(B * T, res_ch, H, W)
                residual = proj(residual)
                residual = residual.reshape(B, T, -1, H, W)
            x = x + residual  # [B, T, out_ch, H, W]
            x = F.relu(x)

        # x: [B, T, hidden_channels[-1], H, W]
        hidden = x

        # === 3. Temporal self-attention ===
        # Reshape: [B, T, hidden, H, W] → [B*H*W, T, hidden]
        C_hid = hidden.shape[2]
        hidden_perm = hidden.permute(0, 3, 4, 1, 2)  # [B, H, W, T, C_hid]
        hidden_flat = hidden_perm.reshape(B * H * W, T, C_hid)

        # Self-attention over T dimension
        attn_out, _ = self.temporal_attention(hidden_flat, hidden_flat, hidden_flat)
        # attn_out: [B*H*W, T, C_hid]

        # Reshape back to [B, H, W, T, C_hid] for per-timestep prediction
        attn_out_reshaped = attn_out.reshape(B, H, W, T, C_hid)  # [B, H, W, T, C_hid]
        return {"attn_out_reshaped": attn_out_reshaped, "hidden": hidden}

    def forward(self, x_dynamic, x_static, land_mask, nan_mask=None):
        """
        Args:
            x_dynamic: [B, T, H, W, C_dyn]
            x_static: [B, H, W, C_stat]
            land_mask: [H, W] or [1, 1, H, W]
            nan_mask: [C_dyn, T, H, W] bool — True where data is valid
        Returns:
            dict with keys: regression, classification, warning, ssta_pred, features
            All predictions preserve [B, T, H, W] or [B, H, W] spatial structure.
        """
        B, T, H, W, C = x_dynamic.shape
        if land_mask is not None and land_mask.dim() == 2:
            land_mask = land_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        feat = self.forward_features(x_dynamic, x_static, land_mask, nan_mask)
        attn_out_reshaped = feat["attn_out_reshaped"]

        # === 4. Prediction heads (per-timestep, per-pixel) ===
        # 每个时间步独立预测: [B, H, W, T, C_hid] → head → [B, H, W, T, 1] → squeeze → [B, H, W, T]
        y_regression = self.regression_head(attn_out_reshaped)  # [B, H, W, T, 1]
        y_regression = y_regression.squeeze(-1).permute(0, 3, 1, 2)  # [B, T, H, W]

        y_classification = self.classification_head(attn_out_reshaped)  # [B, H, W, T, 1]
        y_classification = y_classification.squeeze(-1).permute(0, 3, 1, 2)  # [B, T, H, W]

        y_warning = self.warning_head(attn_out_reshaped)  # [B, H, W, T, 1]
        y_warning = y_warning.squeeze(-1).permute(0, 3, 1, 2)  # [B, T, H, W]

        y_baa = self.baa_head(attn_out_reshaped) * 3.0  # [B, H, W, T, 1] → scale to [0,3]
        y_baa = y_baa.squeeze(-1).permute(0, 3, 1, 2)  # [B, T, H, W]

        # Hotspot 预测: per time step, per spatial location (物理约束用)
        # 取 hidden 前 T 步, 预测每个时空格点的 Hotspot (热应力 ≥ 0)
        hidden = feat["hidden"]  # [B, T, C_hid, H, W]
        hotspot_hidden = hidden[:, :self.T]  # [B, self.T, C_hid, H, W]
        hotspot_input = hotspot_hidden.permute(0, 1, 3, 4, 2)  # [B, self.T, H, W, C_hid]
        hotspot_pred = self.hotspot_head(hotspot_input).squeeze(-1)  # [B, self.T, H, W]

        # Features for explainability: learned weighted aggregation over T
        # attn_out_reshaped: [B, H, W, T, C_hid] → reshape → [B*H*W, T, C_hid]
        C_hid = attn_out_reshaped.shape[-1]
        attn_out_flat = attn_out_reshaped.reshape(B * H * W, T, C_hid)  # [B*H*W, T, C_hid]
        attn_weights = F.softmax(self.temporal_agg(attn_out_flat), dim=1)  # [B*H*W, T, 1]
        features = (attn_weights * attn_out_flat).sum(dim=1)  # [B*H*W, C_hid]
        features = features.reshape(B, H, W, C_hid).permute(0, 3, 1, 2)  # [B, C_hid, H, W]
        features = features.permute(0, 2, 3, 1).mean(dim=(1, 2))  # [B, C_hid]

        return {
            'regression': y_regression,        # [B, T, H, W]
            'classification': y_classification,  # [B, T, H, W]
            'warning': y_warning,              # [B, T, H, W]
            'baa': y_baa,                      # [B, T, H, W]
            'features': features,              # [B, C_hid]
            'hotspot_pred': hotspot_pred,      # [B, T, H, W]
        }