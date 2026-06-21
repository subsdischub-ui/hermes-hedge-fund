"""Probability of Backtest Overfitting via Combinatorial Symmetric CV
(Bailey, Borwein, López de Prado, Zhu). High PBO => the in-sample-best config
tends to underperform out-of-sample => the selection is overfit."""

from hermes.validation.pbo import probability_backtest_overfitting


def test_pbo_zero_for_genuinely_dominant_config():
    # config 0 is always best both in- and out-of-sample -> never overfit.
    matrix = [[0.03, 0.01, -0.02] for _ in range(40)]
    assert probability_backtest_overfitting(matrix, n_splits=10) == 0.0


def test_pbo_bounded_and_high_for_inverted_relationship():
    # IS winner becomes OOS loser in the second half -> overfit -> high PBO.
    first = [[0.05, 0.0] for _ in range(20)]   # config 0 wins early (IS)
    second = [[-0.05, 0.0] for _ in range(20)]  # config 0 loses late (OOS)
    pbo = probability_backtest_overfitting(first + second, n_splits=8)
    assert 0.0 <= pbo <= 1.0
    # Clearly elevated vs the genuine-dominant case (0.0); symmetric inversions
    # cap near 0.5 because balanced splits are neutral ties.
    assert pbo > 0.4


def test_pbo_degenerate_inputs_return_zero():
    assert probability_backtest_overfitting([[0.1]], n_splits=10) == 0.0  # single config
