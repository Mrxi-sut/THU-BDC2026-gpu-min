import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def _feature_group_name(feature_name):
    """Map engineered features to finance-semantic groups used by the gate."""
    name = str(feature_name)
    upper = name.upper()
    lower = name.lower()

    if name == 'instrument':
        return 'identity'
    if name in {'开盘', '收盘', '最高', '最低'} or upper in {'OPEN0', 'HIGH0', 'LOW0', 'VWAP0'}:
        return 'price'
    if name in {'成交量', '成交额', '换手率'}:
        return 'volume_liquidity'
    if name in {'振幅', '涨跌额', '涨跌幅'}:
        return 'return_momentum'
    if upper.startswith(('ROC', 'RANK', 'RSV')) or lower.startswith('return_') or lower in {'rsi', 'macd', 'macd_signal', 'kdj_k', 'kdj_d', 'kdj_j'}:
        return 'return_momentum'
    if upper.startswith(('MA', 'EMA', 'SMA', 'BETA', 'RSQR', 'RESI')) or lower.startswith(('sma_', 'ema_')):
        return 'trend'
    if upper.startswith('STD') or lower.startswith(('volatility_', 'atr_', 'boll_')):
        return 'volatility'
    if upper.startswith(('VMA', 'VSTD', 'WVMA', 'VSUMP', 'VSUMN', 'VSUMD')) or lower.startswith(('volume_', 'obv')):
        return 'volume_liquidity'
    if upper.startswith(('MAX', 'MIN', 'QTLU', 'QTLD', 'IMAX', 'IMIN', 'IMXD')):
        return 'breakout_position'
    if upper.startswith(('KMID', 'KLEN', 'KUP', 'KLOW', 'KSFT')) or lower.endswith('_spread'):
        return 'candlestick'
    if upper.startswith(('CORR', 'CORD')):
        return 'volume_price_relation'
    if upper.startswith(('CNTP', 'CNTN', 'CNTD', 'SUMP', 'SUMN', 'SUMD')):
        return 'path_statistics'
    return 'other'


def build_semantic_feature_groups(feature_names, input_dim):
    if not feature_names or len(feature_names) != input_dim:
        return [('all_features', list(range(input_dim)))]

    group_order = [
        'identity',
        'price',
        'candlestick',
        'return_momentum',
        'trend',
        'volatility',
        'volume_liquidity',
        'volume_price_relation',
        'breakout_position',
        'path_statistics',
        'other',
    ]
    grouped = {name: [] for name in group_order}
    for idx, feature_name in enumerate(feature_names):
        grouped.setdefault(_feature_group_name(feature_name), []).append(idx)
    return [(name, grouped[name]) for name in group_order if grouped.get(name)]


class SemanticRegimeGatedProjection(nn.Module):
    """
    Finance-semantic feature projection.

    Each feature group receives its own projection. A light market-regime encoder
    reads group-level sequence statistics and emits a softmax gate over groups.
    """
    def __init__(self, input_dim, d_model, feature_names, dropout=0.1, regime_dim=16):
        super(SemanticRegimeGatedProjection, self).__init__()
        groups = build_semantic_feature_groups(feature_names, input_dim)
        self.group_names = [name for name, _ in groups]
        self.group_count = len(groups)
        self.group_projections = nn.ModuleList()
        self.regime_projections = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for group_idx, (_, indices) in enumerate(groups):
            index_tensor = torch.tensor(indices, dtype=torch.long)
            self.register_buffer(f'group_indices_{group_idx}', index_tensor)
            group_dim = len(indices)
            self.group_projections.append(nn.Linear(group_dim, d_model))
            self.regime_projections.append(
                nn.Sequential(
                    nn.Linear(group_dim * 3, regime_dim),
                    nn.GELU(),
                )
            )

        self.gate = nn.Sequential(
            nn.Linear(regime_dim * self.group_count, max(regime_dim, self.group_count * 2)),
            nn.GELU(),
            nn.Linear(max(regime_dim, self.group_count * 2), self.group_count),
            nn.Softmax(dim=-1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch*num_stocks, seq_len, feature_dim]
        projected_groups = []
        regime_states = []
        for group_idx in range(self.group_count):
            indices = getattr(self, f'group_indices_{group_idx}')
            group_x = x.index_select(dim=-1, index=indices)
            projected_groups.append(self.group_projections[group_idx](group_x))

            group_mean = group_x.mean(dim=1)
            group_std = group_x.std(dim=1, unbiased=False)
            group_last = group_x[:, -1, :]
            regime_input = torch.cat([group_mean, group_std, group_last], dim=-1)
            regime_states.append(self.regime_projections[group_idx](regime_input))

        regime_state = torch.cat(regime_states, dim=-1)
        gates = self.gate(regime_state)
        stacked = torch.stack(projected_groups, dim=1)
        gated = (stacked * gates[:, :, None, None]).sum(dim=1)
        return self.norm(self.dropout(gated))

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
class CrossStockAttention(nn.Module):
    """股票间交互注意力模块"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossStockAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, stock_features):
        # stock_features: [batch, num_stocks, d_model]
        # 股票间交互：每只股票都关注其他股票的特征
        attended, _ = self.cross_attention(stock_features, stock_features, stock_features)
        output = self.norm(stock_features + self.dropout(attended))
        return output

class FeatureAttention(nn.Module):
    """特征注意力模块"""
    def __init__(self, d_model, dropout=0.1):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch*num_stocks, seq_len, d_model]
        attention_weights = self.attention(x)  # [batch*num_stocks, seq_len, 1]
        attended = torch.sum(x * attention_weights, dim=1)  # [batch*num_stocks, d_model]
        return self.dropout(attended)

class StockTransformer(nn.Module):
    def __init__(self, input_dim, config, num_stocks, emb_dim=16):
        super(StockTransformer, self).__init__()
        self.model_type = 'RankingTransformer'
        self.config = config
        self.num_stocks = num_stocks
        self.use_semantic_gate = bool(config.get('semantic_feature_gate_enabled', True))
        
        # 输入投影层
        if self.use_semantic_gate:
            self.input_proj = SemanticRegimeGatedProjection(
                input_dim=input_dim,
                d_model=config['d_model'],
                feature_names=config.get('feature_columns'),
                dropout=config['dropout'],
                regime_dim=int(config.get('semantic_gate_regime_dim', 16)),
            )
        else:
            self.input_proj = nn.Linear(input_dim, config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'], config['dropout'], config['sequence_length'])
        
        # 时序特征提取
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        
        # 特征注意力
        self.feature_attention = FeatureAttention(config['d_model'], config['dropout'])
        
        # 股票间交互注意力
        self.cross_stock_attention = CrossStockAttention(config['d_model'], config['nhead'], config['dropout'])
        
        # 排序特异性层
        self.ranking_layers = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.LayerNorm(config['d_model'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        # 最终排序分数输出
        self.score_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, src):
        # src: [batch, num_stocks, seq_len, feature_dim]
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        
        # 重塑为 [batch*num_stocks, seq_len, feature_dim]
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        
        # 输入投影和位置编码
        src_proj = self.input_proj(src_reshaped)  # [batch*num_stocks, seq_len, d_model]
        src_proj = self.pos_encoder(src_proj)
        
        # 时序特征提取
        temporal_features = self.temporal_encoder(src_proj)  # [batch*num_stocks, seq_len, d_model]
        
        # 特征注意力聚合
        aggregated_features = self.feature_attention(temporal_features)  # [batch*num_stocks, d_model]
        
        # 重塑回股票维度用于股票间交互
        stock_features = aggregated_features.view(batch_size, num_stocks, -1)  # [batch, num_stocks, d_model]
        
        # 股票间交互注意力
        interactive_features = self.cross_stock_attention(stock_features)  # [batch, num_stocks, d_model]
        
        # 重塑回原形状
        interactive_features = interactive_features.view(batch_size * num_stocks, -1)
        
        # 排序特异性变换
        ranking_features = self.ranking_layers(interactive_features)  # [batch*num_stocks, d_model//2]
        
        # 生成排序分数
        scores = self.score_head(ranking_features)  # [batch*num_stocks, 1]
        
        # 重塑为最终输出格式
        output = scores.view(batch_size, num_stocks)  # [batch, num_stocks]
        
        return output
