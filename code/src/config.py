# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    'seed': 20260416,
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 5,        # 交易日样本增多后 50 轮容易吃满内存；增强版主要依靠表格融合分支提分
    'learning_rate': 1e-5,  # 稍微降低学习率
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,
    # Transformer输入侧的金融语义分组门控。训练/推理时会把实际
    # feature_columns 写入 config，模型据此按价格、动量、波动率、
    # 成交量、K线、突破等组动态分配权重。
    'semantic_feature_gate_enabled': True,
    'semantic_gate_regime_dim': 16,

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）
    # 轻量版 Decision-Focused Learning：在排序损失外，额外奖励模型当前
    # Top-N 软组合的未来收益。设为 0.0 可完全回到原始训练目标。
    'decision_loss_weight': 0.1,
    'decision_temperature': 0.25,
    'decision_top_k': 10,

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',

    # 提交仓位策略：正式路径只在 Top2/Top3 间切换，禁止 Top1 梭哈，不切到 Top5。
    'dynamic_positioning_enabled': True,

    # 固定仓位备用策略：动态仓位关闭时默认三股进攻，Top5 只在回测里做对照组。
    'prediction_top_k': 3,
    'prediction_weights': [0.4, 0.35, 0.25],
    'attack_position_levels': {
        'strong': 1.0,
        'mid': 0.8,
        'weak': 0.6,
        'very_weak': 0.4,
    },

    # 表格排序增强分支：只使用现有离线日线字段，不改爬虫。
    # 它学习“当日横截面谁更强”，再与 Transformer 分数融合，思路来自月月星队伍的传统模型融合经验。
    'tabular_ranker_enabled': True,
    # 5000积分 guarded production：若存在 no_factors/index_weight 双路表格模型，
    # 预测时按市场分型选择 lane；缺少双路模型时自动回退到单表格模型。
    'tabular_factor_lane_enabled': True,
    'tabular_blend_weight': 0.95,
    # 离线日线最终整合：用二层 selector 在 hgb_rank 稳定性和热点/共识分支之间动态融合。
    'submission_score_col': 'selector_score',
    'tabular_component_weights': {
        'lgb_rank': 0.30,
        'lgb_return': 0.08,
        'hgb_return': 0.16,
        'hgb_rank': 0.14,
        'extra_rank': 0.10,
        'heuristic': 0.22,
    },
    'tabular_heuristic_weights': {
        'hotspot_strength_rank': 0.24,
        'relative_strength_20_rank': 0.16,
        'ret_20_rank': 0.14,
        'ret_10_rank': 0.10,
        'ret_5_rank': 0.08,
        'amount_ratio_20_rank': 0.08,
        'volume_ratio_20_rank': 0.06,
        'turnover_accel_5_20_rank': 0.06,
        'breakout_20_rank': 0.05,
        'board_hotspot_rank': 0.03,
        'amount_moderate_20_rank': 0.00,
        'no_overheat_20_rank': 0.00,
    },
    'tabular_val_days': 40,
    'tabular_embargo_days': 5,
    'tabular_min_history_days': 60,
}
