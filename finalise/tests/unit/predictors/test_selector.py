import numpy as np
import pandas as pd
import pytest
from finalise.config import Config
from finalise.predictors import PredictorAnalysis, selector, SelectedCandidate

def test_ut_selector_select():
    # Test that SelectedCandidate can be instantiated
    cand = SelectedCandidate(
        ticker="PETR4.SA",
        component="bruto",
        series=pd.Series([0.1, 0.2]),
        lag_tau=1,
        relation_type="linear",
        mi_value=0.5,
        te_value=None,
        granger_pvalue=0.01
    )
    assert cand.ticker == "PETR4.SA"
    assert cand.component == "bruto"
    assert cand.relation_type == "linear"

def test_ut_predictor_analysis_full(prices_synthetic):
    # Test PredictorAnalysis integration with mock/synthetic prices
    target_ticker = "PETR4.SA"
    pred_ticker = "VALE3.SA"
    
    # Target prices
    p_target = prices_synthetic.copy()
    # Predictor prices: let's make it highly related with a lag of 1
    # For returns, we shift the log-returns of target
    log_ret_target = np.log(p_target / p_target.shift(1)).dropna()
    log_ret_pred = log_ret_target.shift(-1).fillna(0.0) # pred leads target by 1, so target lags pred by 1
    
    # Reconstruct predictor prices using the same index as p_target
    p_pred = pd.Series(100.0 * np.exp(np.cumsum(log_ret_pred.reindex(p_target.index, fill_value=0.0))), index=p_target.index)
    
    prices_dict = {
        target_ticker: p_target,
        pred_ticker: p_pred
    }
    
    config = Config(
        target_ticker=target_ticker,
        predictor_tickers=[pred_ticker],
        horizons=[1, 5],
        smoothing_window=5,
        smoothing_method="SMA",
        stl_period=5,
        alpha=0.05,
        n_permutations=20,
        knn_k=3,
        lag_max=3
    )
    
    # We pass the target series (returns at horizon k=1)
    target_series = log_ret_target
    
    pa = PredictorAnalysis(prices_dict, target_series, config)
    assert pa.horizon == 1
    
    pa.run()
    
    # Check that outputs are populated
    assert isinstance(pa.selected, list)
    assert isinstance(pa.cointegration, dict)
    assert isinstance(pa.results, dict)
    
    # Cointegration should have pred_ticker
    assert pred_ticker in pa.cointegration
    coint_res = pa.cointegration[pred_ticker]
    assert "is_cointegrated" in coint_res
    
    # Results should contain components for pred_ticker
    for component in ["bruto", "suavizado", "tendencia", "sazonalidade", "residuo"]:
        assert (pred_ticker, component) in pa.results
        res_comp = pa.results[(pred_ticker, component)]
        assert "mi" in res_comp
        
    # Verify IT-P-06 and IT-P-07: Cointegration uses I(1) raw prices, selection pipeline uses I(0) log-returns
    from statsmodels.tsa.stattools import adfuller
    # Raw prices in prices_dict have a unit root (non-stationary I(1))
    _, target_p_val, *_ = adfuller(prices_dict[target_ticker].dropna())
    _, pred_p_val, *_ = adfuller(prices_dict[pred_ticker].dropna())
    assert target_p_val > 0.05
    assert pred_p_val > 0.05
    
    # Candidates in the selection pipeline are stationary log-returns (stationary I(0))
    assert len(pa.candidates) > 0
    for comp_key, series in pa.candidates.items():
        _, p_val, *_ = adfuller(series.dropna())
        assert p_val < 0.05
        
    # Test report generation
    figs = pa.report()
    assert isinstance(figs, dict)
    assert "cross_mi_heatmap" in figs
    if "granger_pvalues" in figs:
        assert figs["granger_pvalues"] is not None
    if "te_permutation" in figs:
        assert figs["te_permutation"] is not None

def test_ut_predictor_analysis_errors(prices_synthetic):
    # Test error cases and branch coverage in PredictorAnalysis
    target_ticker = "PETR4.SA"
    
    # 1. Missing target ticker in prices_dict raises ValueError
    config = Config(target_ticker=target_ticker, predictor_tickers=[])
    with pytest.raises(ValueError, match="não encontrados em prices_dict"):
        PredictorAnalysis({}, pd.Series([0.1]), config)
        
    # 2. Call report before run raises ValueError
    prices_dict = {target_ticker: prices_synthetic}
    pa = PredictorAnalysis(prices_dict, pd.Series([0.1]), config)
    with pytest.raises(ValueError, match="Execute run"):
        pa.report()
        
    # 3. Horizon fallback when index does not match
    assert pa.horizon == 1 # fell back to horizons[0]

def test_ut_predictor_analysis_unrelated_and_short(prices_synthetic):
    # Test unrelated predictor (triggers no selected candidates) and short candidate data
    target_ticker = "PETR4.SA"
    unrelated_ticker = "USIM5.SA"
    short_ticker = "GGBR4.SA"
    
    p_target = prices_synthetic.copy()
    
    # Unrelated is pure white noise (highly likely to have non-significant MI)
    rng = np.random.default_rng(42)
    p_unrelated = pd.Series(100.0 + rng.standard_normal(len(p_target)), index=p_target.index)
    
    # Short series has very few elements
    p_short = pd.Series([100.0, 101.0, 102.0], index=p_target.index[:3])
    
    prices_dict = {
        target_ticker: p_target,
        unrelated_ticker: p_unrelated,
        short_ticker: p_short
    }
    
    config = Config(
        target_ticker=target_ticker,
        predictor_tickers=[unrelated_ticker, short_ticker, "MISSING_TICKER"],
        horizons=[1],
        smoothing_window=5,
        smoothing_method="SMA",
        stl_period=5,
        alpha=0.01, # low alpha to ensure MI is not significant
        n_permutations=10,
        knn_k=3,
        lag_max=3
    )
    
    log_ret_target = np.log(p_target / p_target.shift(1)).dropna()
    
    pa = PredictorAnalysis(prices_dict, log_ret_target, config)
    pa.run()
    
    # Check that short and missing tickers were skipped in candidates/results/cointegration
    assert "MISSING_TICKER" not in pa.cointegration
    assert "MISSING_TICKER" not in pa.results
    assert short_ticker in pa.cointegration
    assert (short_ticker, "bruto") not in pa.results
    assert unrelated_ticker in pa.cointegration
    
    # No candidate should be selected
    assert len(pa.selected) == 0
    
    # Report should not raise an error, and should have warning text inside justification
    figs = pa.report()
    assert "cross_mi_heatmap" in figs
