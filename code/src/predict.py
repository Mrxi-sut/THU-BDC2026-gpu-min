import os
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from model import StockTransformer
from positioning import (
	choose_factor_lane_v2,
	choose_factor_lane_v3,
	choose_positioning,
	compute_lane_overlap_features,
	compute_market_snapshot,
	compute_score_profile,
	select_portfolio_from_ranking,
)
from tabular_ranker import SELECTOR_AUXILIARY_PREFIXES, add_strategy_selector_score, predict_tabular_ranker
from utils import engineer_features_39, engineer_features_158plus39


feature_cloums_map = {
	'39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	],
	'158+39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
		'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
		'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
		'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
		'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
		'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
		'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
		'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
		'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
		'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
		'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
		'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
		'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
		'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
		'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	]
}

feature_engineer_func_map = {
	'39': engineer_features_39,
	'158+39': engineer_features_158plus39,
}


def rank_unit(values):
	"""把不同来源的模型分数转成 0~1 排名分，避免量纲不一致。"""
	series = pd.Series(values)
	return series.rank(pct=True, method='average').fillna(0.5).to_numpy(dtype=np.float32)


def preprocess_predict_data(df, stockid2idx):
	assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
	feature_engineer = feature_engineer_func_map[config['feature_num']]
	feature_columns = feature_cloums_map[config['feature_num']]

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(10, mp.cpu_count())
	print('cpus!!!!!!!!!!!!!!!!!!',mp.cpu_count())
	with mp.Pool(processes=num_processes) as pool:
		processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

	processed = pd.concat(processed_list).reset_index(drop=True)
	processed['instrument'] = processed['股票代码'].map(stockid2idx)
	processed = processed.dropna(subset=['instrument']).copy()
	processed['instrument'] = processed['instrument'].astype(np.int64)
	processed['日期'] = pd.to_datetime(processed['日期'])

	return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
	sequences, sequence_stock_ids = [], []
	for stock_id in stock_ids:
		stock_history = data[
			(data['股票代码'] == stock_id) &
			(data['日期'] <= latest_date)
		].sort_values('日期').tail(sequence_length)

		if len(stock_history) == sequence_length:
			sequences.append(stock_history[features].values.astype(np.float32))
			sequence_stock_ids.append(stock_id)

	if len(sequences) == 0:
		raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def main():
	data_file = os.environ.get('PREDICT_DATA_FILE', os.path.join(config['data_path'], 'train.csv'))
	model_path = os.path.join(config['output_dir'], 'best_model.pth')
	scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
	output_path = os.environ.get('PREDICT_OUTPUT_PATH', os.path.join('./output/', 'result.csv'))

	if not os.path.exists(model_path):
		raise FileNotFoundError(f'未找到模型文件: {model_path}')
	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

	raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'])
	latest_date = raw_df['日期'].max()

	stock_ids = sorted(raw_df['股票代码'].unique())
	stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

	processed, features = preprocess_predict_data(raw_df, stockid2idx)
	config['feature_columns'] = list(features)
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

	scaler = joblib.load(scaler_path)
	processed[features] = scaler.transform(processed[features])

	sequence_length = config['sequence_length']
	sequences_np, sequence_stock_ids = build_inference_sequences(
		processed,
		features,
		sequence_length,
		stock_ids,
		latest_date,
	)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.to(device)
	model.eval()

	with torch.no_grad():
		x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]
		scores = model(x).squeeze(0).detach().cpu().numpy()         # [N]

	score_df = pd.DataFrame({
		'stock_id': sequence_stock_ids,
		'transformer_score': scores,
	})
	score_df['transformer_rank'] = rank_unit(score_df['transformer_score'])
	submission_score_col = config.get('submission_score_col')
	positioning_snapshot = None
	second_slot_challenger_df = None
	second_slot_challenger_score_col = None

	def build_tabular_lane_prediction(model_file_name, factor_policy, lane_name):
		lane_pred = predict_tabular_ranker(
			raw_df,
			config['output_dir'],
			model_file_name=model_file_name,
			factor_policy=factor_policy,
		)
		snapshot_seed_col = 'controlled_hotspot' if 'controlled_hotspot' in lane_pred.columns else 'selector_score'
		if snapshot_seed_col in lane_pred.columns:
			snapshot_seed = lane_pred.sort_values(snapshot_seed_col, ascending=False)['股票代码'].head(5).tolist()
		else:
			snapshot_seed = lane_pred['股票代码'].head(5).tolist()
		lane_snapshot = compute_market_snapshot(raw_df, latest_date, snapshot_seed)
		lane_pred = add_strategy_selector_score(lane_pred, lane_snapshot)
		if 'selector_score' in lane_pred.columns:
			profile_df = lane_pred.sort_values('selector_score', ascending=False).rename(columns={'selector_score': 'final_score'})
			profile = compute_score_profile(profile_df, 'final_score')
		else:
			profile_df = lane_pred.sort_values('tabular_score', ascending=False).rename(columns={'tabular_score': 'final_score'})
			profile = compute_score_profile(profile_df, 'final_score')
		lane_snapshot['factor_lane_candidate'] = lane_name
		return lane_pred, lane_snapshot, profile

	def lane_debug_row(lane_name, lane_snapshot, lane_profile, lane_pred):
		score_col = 'selector_score' if 'selector_score' in lane_pred.columns else 'tabular_score'
		top_codes = lane_pred.sort_values(score_col, ascending=False)['股票代码'].astype(str).str.zfill(6).head(5).tolist()
		row = {
			'lane': lane_name,
			'top5': ','.join(top_codes),
			'market_regime_v2': lane_snapshot.get('market_regime_v2', ''),
			'selector_regime': lane_pred.sort_values(score_col, ascending=False).iloc[0].get('selector_regime', ''),
			'market_ret_20': lane_snapshot.get('market_ret_20', np.nan),
			'up_ratio_20': lane_snapshot.get('up_ratio_20', np.nan),
			'selected_ret_20_mean': lane_snapshot.get('selected_ret_20_mean', np.nan),
		}
		for key in [
			'top1_top2_gap', 'top2_top3_gap', 'top2_vs_top10_gap',
			'top2_concentration', 'top3_concentration',
			'top2_relative_strength_20_rank', 'top2_hotspot_strength_rank',
			'top2_controlled_hotspot', 'top2_selector_consensus',
		]:
			row[key] = lane_profile.get(key, np.nan)
		return row

	tabular_model_path = os.path.join(config['output_dir'], 'tabular_ranker.pkl')
	lane_index_path = os.path.join(config['output_dir'], 'tabular_ranker_index_weight.pkl')
	lane_baseline_path = os.path.join(config['output_dir'], 'tabular_ranker_no_factors.pkl')
	lane_industry_core_path = os.path.join(config['output_dir'], 'tabular_ranker_index_industry_core.pkl')
	use_factor_lane = (
		config.get('tabular_factor_lane_enabled', True)
		and os.path.exists(lane_index_path)
		and os.path.exists(lane_baseline_path)
	)
	if config.get('tabular_ranker_enabled', True) and (os.path.exists(tabular_model_path) or use_factor_lane):
		# 增强模型只使用 train.csv 最后一日之前的信息；这里做分数融合，不读取 test.csv。
		selected_lane = 'single_model'
		if use_factor_lane:
			index_pred, index_snapshot, index_profile = build_tabular_lane_prediction(
				'tabular_ranker_index_weight.pkl',
				'first_tier',
				'index_weight',
			)
			no_factor_pred, no_factor_snapshot, no_factor_profile = build_tabular_lane_prediction(
				'tabular_ranker_no_factors.pkl',
				'none',
				'no_factors',
			)
			lane_overlap = compute_lane_overlap_features(index_pred, no_factor_pred)
			lane_pred_map = {
				'index_weight': index_pred,
				'no_factors': no_factor_pred,
			}
			lane_audit_rows = [
				lane_debug_row('index_weight', index_snapshot, index_profile, index_pred),
				lane_debug_row('no_factors', no_factor_snapshot, no_factor_profile, no_factor_pred),
			]
			if (
				config.get('tabular_industry_core_lane_enabled', True)
				and os.path.exists(lane_industry_core_path)
			):
				industry_pred, industry_snapshot, industry_profile = build_tabular_lane_prediction(
					'tabular_ranker_index_industry_core.pkl',
					'index_weight,industry_core',
					'index_industry_core',
				)
				industry_overlap = compute_lane_overlap_features(industry_pred, no_factor_pred)
				second_slot_challenger_df = industry_pred.rename(columns={'股票代码': 'stock_id'}).copy()
				second_slot_challenger_score_col = 'selector_score' if 'selector_score' in second_slot_challenger_df.columns else 'tabular_score'
				selected_lane, lane_snapshot = choose_factor_lane_v3(
					index_snapshot,
					index_profile,
					no_factor_snapshot,
					no_factor_profile,
					industry_snapshot,
					industry_profile,
					lane_overlap,
					industry_overlap,
					'index_industry_core',
				)
				lane_pred_map['index_industry_core'] = industry_pred
				lane_audit_rows.append(lane_debug_row('index_industry_core', industry_snapshot, industry_profile, industry_pred))
			else:
				selected_lane, lane_snapshot = choose_factor_lane_v2(
					index_snapshot,
					index_profile,
					no_factor_snapshot,
					no_factor_profile,
					lane_overlap,
				)
			tabular_pred = lane_pred_map.get(selected_lane, no_factor_pred)
			positioning_snapshot = dict(lane_snapshot)
			for row in lane_audit_rows:
				if row['lane'] == 'index_weight':
					row['market_regime_v2'] = lane_snapshot.get('index_market_regime_v2', row.get('market_regime_v2', ''))
				if row['lane'] == 'no_factors':
					row['market_regime_v2'] = lane_snapshot.get('nofactor_market_regime_v2', row.get('market_regime_v2', ''))
				if row['lane'] == 'index_industry_core':
					row['market_regime_v2'] = lane_snapshot.get('industry_market_regime_v2', row.get('market_regime_v2', ''))
				row.update({
					'chosen_lane': selected_lane,
					'factor_lane_reason': lane_snapshot.get('factor_lane_reason', ''),
					'lane_top2_overlap': lane_snapshot.get('lane_top2_overlap', np.nan),
					'lane_top5_overlap': lane_snapshot.get('lane_top5_overlap', np.nan),
					'industry_challenger_action': lane_snapshot.get('industry_challenger_action', ''),
					'industry_nofactor_top2_overlap': lane_snapshot.get('industry_nofactor_top2_overlap', np.nan),
				})
			pd.DataFrame(lane_audit_rows).to_csv(os.path.join('./output/', 'lane_audit.csv'), index=False)
			print(
				"已启用守门多路表格模型: "
				f"lane={selected_lane}, reason={positioning_snapshot.get('factor_lane_reason', '')}"
			)
		else:
			tabular_pred = predict_tabular_ranker(raw_df, config['output_dir'])
			snapshot_seed_col = 'controlled_hotspot' if 'controlled_hotspot' in tabular_pred.columns else 'selector_score'
			snapshot_seed = tabular_pred.sort_values(snapshot_seed_col, ascending=False)['股票代码'].head(5).tolist() if snapshot_seed_col in tabular_pred.columns else tabular_pred['股票代码'].head(5).tolist()
			selector_snapshot = compute_market_snapshot(raw_df, latest_date, snapshot_seed)
			positioning_snapshot = dict(selector_snapshot)
			tabular_pred = add_strategy_selector_score(tabular_pred, selector_snapshot)
		tabular_pred = tabular_pred.rename(columns={'股票代码': 'stock_id'})
		tabular_cols = [
			col for col in [
				'stock_id', 'tabular_score', 'lgb_rank', 'lgb_return',
				'hgb_return', 'hgb_rank', 'extra_rank', 'heuristic',
				'hotspot_strength_rank', 'relative_strength_20_rank',
				'board_hotspot_rank', 'controlled_hotspot',
				'amount_moderate_20_rank', 'no_overheat_20_rank',
				'selector_score', 'selector_consensus',
				'selector_hot_rotation', 'selector_defensive', 'selector_regime',
				'selector_hgb_overlap',
			]
			if col in tabular_pred.columns
		]
		for col in tabular_pred.columns:
			if col.startswith(SELECTOR_AUXILIARY_PREFIXES) and col not in tabular_cols:
				tabular_cols.append(col)
		score_df = score_df.merge(
			tabular_pred[tabular_cols],
			on='stock_id',
			how='left',
		)
		score_df['tabular_score'] = score_df['tabular_score'].fillna(score_df['tabular_score'].median()).fillna(0.5)
		score_df['tabular_rank'] = rank_unit(score_df['tabular_score'])
		tabular_weight = config.get('tabular_blend_weight', 0.85)
		transformer_weight = 1.0 - tabular_weight
		score_df['final_score'] = (
			tabular_weight * score_df['tabular_rank']
			+ transformer_weight * score_df['transformer_rank']
		)
		if submission_score_col and submission_score_col in score_df.columns:
			# 滚动回测选择的最终提交主信号：按当日横截面排名输出 Top2/Top3。
			score_df['final_score'] = rank_unit(score_df[submission_score_col])
			print(f'最终提交排序使用主信号: {submission_score_col}')
		if not use_factor_lane:
			print(f'已启用表格排序增强模型: {tabular_model_path}')
	else:
		score_df['final_score'] = score_df['transformer_rank']
		print('未找到表格排序增强模型，使用 Transformer 原始排序。')

	score_df = score_df.sort_values('final_score', ascending=False).reset_index(drop=True)
	ranked_stock_ids = score_df['stock_id'].tolist()

	if config.get('dynamic_positioning_enabled', True):
		# 只用train.csv中预测日之前的行情判断市场环境，不联网、不读取test。
		if positioning_snapshot is not None:
			snapshot = dict(positioning_snapshot)
		else:
			snapshot = compute_market_snapshot(raw_df, latest_date, ranked_stock_ids[:5])
		if 'selector_regime' in score_df.columns and len(score_df) > 0:
			snapshot['selector_regime'] = str(score_df.iloc[0]['selector_regime'])
		profile_col = submission_score_col if submission_score_col and submission_score_col in score_df.columns else 'final_score'
		score_profile = compute_score_profile(score_df, profile_col)
		position_name, prediction_weights, snapshot = choose_positioning(snapshot, len(ranked_stock_ids), score_profile)
		selected_stock_ids, prediction_weights, snapshot = select_portfolio_from_ranking(
			score_df,
			prediction_weights,
			snapshot,
			score_profile,
			score_col='final_score',
			challenger_df=second_slot_challenger_df,
			challenger_score_col=second_slot_challenger_score_col,
			challenger_name='index_industry_core',
		)
		prediction_top_k = len(prediction_weights)
	else:
		position_name = 'fixed_config'
		snapshot = {}
		prediction_top_k = int(config.get('prediction_top_k', 5))
		prediction_weights = list(config.get('prediction_weights', [1.0 / prediction_top_k] * prediction_top_k))
		selected_stock_ids = ranked_stock_ids[:prediction_top_k]

	# 比赛规则最多5支股票，权重和不超过1。
	if not 1 <= prediction_top_k <= 5:
		raise ValueError(f'prediction_top_k 必须在1到5之间，当前为 {prediction_top_k}')
	if len(prediction_weights) != prediction_top_k:
		raise ValueError('prediction_weights 的数量必须等于 prediction_top_k')
	if sum(prediction_weights) > 1.0 + 1e-8:
		raise ValueError(f'预测权重和不能超过1，当前为 {sum(prediction_weights):.6f}')
	if len(ranked_stock_ids) < prediction_top_k:
		raise ValueError(f'可预测股票不足{prediction_top_k}只，当前仅有 {len(ranked_stock_ids)} 只')

	output_df = pd.DataFrame({
		'stock_id': selected_stock_ids,
		'weight': prediction_weights,
	})
	output_df.to_csv(output_path, index=False, encoding='utf-8', lineterminator='\n')
	score_df.to_csv(os.path.join('./output/', 'ranking_debug.csv'), index=False)
	audit_items = {
		'latest_date': latest_date.date().isoformat(),
		'position_name': position_name,
		'weights': '/'.join(f'{w:.2f}' for w in prediction_weights),
		'total_weight': sum(prediction_weights),
		'selected_stock_ids': ','.join(selected_stock_ids),
	}
	for key in [
		'positioning_policy', 'selector_regime', 'topk_reason', 'p_top2_rule',
		'market_regime_v2', 'factor_lane', 'factor_lane_reason',
		'index_market_regime_v2', 'nofactor_market_regime_v2', 'industry_market_regime_v2',
		'lane_top2_overlap', 'lane_top5_overlap',
		'index_top2_quality', 'nofactor_top2_quality',
		'industry_challenger_action', 'industry_challenger_reason',
		'industry_nofactor_top2_overlap', 'industry_nofactor_top5_overlap',
		'industry_top2_quality', 'industry_top2_vs_top10_gap',
		'weak_spike_top2_flag', 'third_tail_risk_flag',
		'third_overheat_tail_flag', 'third_unsupported_tail_flag',
		'second_quality', 'third_quality', 'top2_weight_reason',
		'portfolio_selection_action', 'second_slot_risky_flag',
		'second_slot_replacement_code', 'second_slot_replacement_reason',
		'second_industry_confirmation', 'third_industry_confirmation',
		'second_overheat_risk', 'third_overheat_risk',
		'signal_weak_flag', 'candidate_hot_flag', 'high_dispersion_flag',
		'market_ret_5', 'market_ret_20', 'up_ratio_5', 'up_ratio_20',
		'market_drawdown_20', 'selected_ret_5_mean', 'selected_ret_20_mean',
		'score_top2_top3_gap', 'score_top2_vs_top10_gap',
		'score_top3_no_overheat_20_rank', 'score_top3_amount_moderate_20_rank',
		'score_top3_hotspot_strength_rank', 'score_top3_relative_strength_20_rank',
	]:
		if key in snapshot:
			audit_items[key] = snapshot[key]
	pd.DataFrame([audit_items]).to_csv(os.path.join('./output/', 'positioning_audit.csv'), index=False)

	print(f'预测日期: {latest_date.date()}')
	print(f'参与排序股票数: {len(ranked_stock_ids)}')
	print(f'仓位策略: {position_name}, 权重: {prediction_weights}, 总仓位: {sum(prediction_weights):.2f}')
	if snapshot.get('topk_reason'):
		print(f"仓位原因: {snapshot.get('topk_reason')}")
	if snapshot:
		important_keys = [
			'market_ret_5', 'market_ret_20', 'up_ratio_5', 'up_ratio_20',
			'selected_ret_20_mean', 'market_drawdown_20'
		]
		snapshot_text = ', '.join(
			f'{key}={snapshot[key]:.4f}' for key in important_keys if key in snapshot
		)
		print(f'市场状态: {snapshot_text}')
	print(f'Top{prediction_top_k}融合明细:')
	print(score_df.head(prediction_top_k).to_string(index=False))
	print(f'结果已写入: {output_path}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()
