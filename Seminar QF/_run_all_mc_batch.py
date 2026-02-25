from pathlib import Path
import pandas as pd
from datetime import datetime

from src.analysis.monte_carlo_garch import monte_carlo_garch_1year_parallel
from src.analysis.monte_carlo_regime_switching import monte_carlo_regime_switching_1year_parallel
from src.analysis.monte_carlo_ms_garch import monte_carlo_ms_garch_1year_parallel
from src.analysis.monte_carlo_merton import monte_carlo_merton_1year_parallel

root = Path('.')
out_dir = root / 'batch_results'
out_dir.mkdir(parents=True, exist_ok=True)

num_simulations = 1000
num_days = 1260
n_jobs = -1

files = {
    'garch': root / 'data/output/daily_asset_returns_with_garch.csv',
    'regime': root / 'data/output/daily_asset_returns_with_regime_switching.csv',
    'regime_fallback': root / 'data/output/daily_asset_returns_with_regime.csv',
    'ms_garch': root / 'data/output/daily_asset_returns_with_msgarch.csv',
    'merton': root / 'data/output/merged_data_with_merton.csv',
    'cds_filter': root / 'data/cds_filters/gvkey_maturity_simulation_windows.csv',
}

regime_file = files['regime'] if files['regime'].exists() else files['regime_fallback']
merton_file = files['merton']
cds_filter_file = files['cds_filter']
ts = datetime.now().strftime('%Y%m%d_%H%M%S')

print('Running all MC models with simulations={}, days={}, n_jobs={}'.format(num_simulations, num_days, n_jobs))
print('Using regime file: {}'.format(regime_file))
print('Using CDS date filter file: {}'.format(cds_filter_file))

results_paths = {}

print('\n[1/4] GARCH...')
garch_df = monte_carlo_garch_1year_parallel(
    garch_file=str(files['garch']),
    merton_file=str(merton_file),
    num_simulations=num_simulations,
    num_days=num_days,
    n_jobs=n_jobs,
    cds_filter_file=str(cds_filter_file),
)
results_paths['garch'] = out_dir / 'batch_results_garch_{}_{}.csv'.format(num_simulations, ts)
garch_df.to_csv(results_paths['garch'], index=False)
print('Saved: {} ({:,} rows)'.format(results_paths['garch'], len(garch_df)))

print('\n[2/4] Regime Switching...')
rs_df = monte_carlo_regime_switching_1year_parallel(
    regime_params_file=str(regime_file),
    merton_file=str(merton_file),
    num_simulations=num_simulations,
    num_days=num_days,
    n_jobs=n_jobs,
    cds_filter_file=str(cds_filter_file),
)
results_paths['regime_switching'] = out_dir / 'batch_results_regime_switching_{}_{}.csv'.format(num_simulations, ts)
rs_df.to_csv(results_paths['regime_switching'], index=False)
print('Saved: {} ({:,} rows)'.format(results_paths['regime_switching'], len(rs_df)))

print('\n[3/4] MS-GARCH...')
msg_df = monte_carlo_ms_garch_1year_parallel(
    ms_garch_file=str(files['ms_garch']),
    merton_file=str(merton_file),
    num_simulations=num_simulations,
    num_days=num_days,
    n_jobs=n_jobs,
    cds_filter_file=str(cds_filter_file),
)
results_paths['ms_garch'] = out_dir / 'batch_results_ms_garch_{}_{}.csv'.format(num_simulations, ts)
msg_df.to_csv(results_paths['ms_garch'], index=False)
print('Saved: {} ({:,} rows)'.format(results_paths['ms_garch'], len(msg_df)))

print('\n[4/4] Merton...')
merton_df = monte_carlo_merton_1year_parallel(
    merton_file=str(merton_file),
    num_simulations=num_simulations,
    num_days=num_days,
    n_jobs=n_jobs,
    cds_filter_file=str(cds_filter_file),
)
results_paths['merton'] = out_dir / 'batch_results_merton_{}_{}.csv'.format(num_simulations, ts)
merton_df.to_csv(results_paths['merton'], index=False)
print('Saved: {} ({:,} rows)'.format(results_paths['merton'], len(merton_df)))

summary = pd.DataFrame([
    {'model': 'garch', 'rows': len(garch_df), 'file': str(results_paths['garch'])},
    {'model': 'regime_switching', 'rows': len(rs_df), 'file': str(results_paths['regime_switching'])},
    {'model': 'ms_garch', 'rows': len(msg_df), 'file': str(results_paths['ms_garch'])},
    {'model': 'merton', 'rows': len(merton_df), 'file': str(results_paths['merton'])},
])
summary_path = out_dir / 'batch_results_summary_{}_{}.csv'.format(num_simulations, ts)
summary.to_csv(summary_path, index=False)
print('\n=== COMPLETE ===')
print(summary.to_string(index=False))
print('\nSummary file: {}'.format(summary_path))
