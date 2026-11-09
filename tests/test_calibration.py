"""Tests for :mod:`src.engine.calibration`: the correction is learnt out of sample and used only when it helps.

The frames are synthetic with known truth: outcomes drawn from true
probabilities, then shown to the calibrator either as they are (nothing to
learn: the identity must be kept) or deliberately distorted (overconfident,
biased: the correction must find it and improve the later matches).
"""
from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from src.common.schemas import Fixture, Market, MatchPrediction
from src.engine import backtest as bt
from src.engine import calibration as cal
from src.engine.model import dixon_coles_pmf, markets_from_matrix
from tests.test_backtest import make_league


def truth_frame(n: int, *, seed: int, families: tuple[str, ...] = ("Serie C", "Serie D"), start: date = date(2025, 8, 1)) -> pd.DataFrame:
    """``n`` matches over ``n // 40`` days with true Dixon-Coles (``rho = 0``) probabilities and sampled scores."""
    rng = np.random.default_rng(seed)
    lam, mu = rng.uniform(0.5, 2.6, n), rng.uniform(0.4, 2.0, n)
    hg, ag = rng.poisson(lam), rng.poisson(mu)
    days = pd.to_datetime(start) + pd.to_timedelta(np.sort(rng.integers(0, max(n // 40, 2), n)), unit="D")
    frame = pd.DataFrame(
        {
            "match_id": [f"M{i:05d}" for i in range(n)],
            "date": days,
            "family": rng.choice(list(families), n),
            "girone": "A",
            "section": "S",
            "home_goals": hg,
            "away_goals": ag,
            "lam": lam,
            "mu": mu,
            "rho": 0.0,
            **bt._market_arrays(lam, mu, 0.0, 10),
        }
    )
    total = hg + ag
    return frame.assign(
        res_1X2=np.where(hg > ag, "1", np.where(hg == ag, "X", "2")),
        gg=((hg > 0) & (ag > 0)).astype(int),
        o15=(total > 1.5).astype(int),
        o25=(total > 2.5).astype(int),
        o35=(total > 3.5).astype(int),
    )


def overconfident(frame: pd.DataFrame, *, power: float = 2.2, gg_shift: float = 0.5, gg_slope: float = 1.8) -> pd.DataFrame:
    """Sharpen 1X2 (``p ** power`` renormalised) and distort GG (``logit p' = shift + slope * logit p``)."""
    out = frame.copy()
    probs = out[["p_1", "p_X", "p_2"]].to_numpy() ** power
    out[["p_1", "p_X", "p_2"]] = probs / probs.sum(axis=1, keepdims=True)
    gg = expit(gg_shift + gg_slope * logit(np.clip(out["p_GG"].to_numpy(), 1e-6, 1 - 1e-6)))
    return out.assign(p_GG=gg, p_NG=1 - gg)


@pytest.fixture(scope="module")
def calibrated() -> pd.DataFrame:
    return truth_frame(6000, seed=1)


# ============================================================================ the verdict
def test_identity_is_kept_when_there_is_nothing_to_gain(calibrated: pd.DataFrame) -> None:
    model = cal.fit_recalibration(calibrated)
    assert not model.enabled and model.enabled_markets == ()
    assert set(model.markets) == set(bt.DEFAULT_MARKETS)
    for calibration in model.markets.values():
        assert calibration.n_eval >= 1500 and calibration.gain < model.min_gain and "sotto la soglia" in calibration.reason
        assert calibration.pooled.is_identity
    assert model.describe().startswith("Calibrazione applicata: no") and "probabilità originali" in model.describe()
    probabilities = {"1X2": {"1": 0.5, "X": 0.3, "2": 0.2}, Market.BTTS: {"GG": 0.55, "NG": 0.45}}
    assert model.apply(probabilities) == {"1X2": {"1": 0.5, "X": 0.3, "2": 0.2}, Market.BTTS: {"GG": 0.55, "NG": 0.45}}
    pd.testing.assert_frame_equal(model.apply_frame(calibrated), calibrated)


def test_a_miscalibrated_forecast_is_corrected_and_the_gain_holds_on_new_matches(calibrated: pd.DataFrame) -> None:
    shown = overconfident(calibrated)
    model = cal.fit_recalibration(shown)
    assert model.enabled and set(model.enabled_markets) == {Market.MATCH_ODDS, Market.BTTS}
    assert model.markets[Market.MATCH_ODDS].gain > 0.02 and model.markets[Market.BTTS].gain > 0.02
    assert not model.markets[Market.OVER_UNDER_25].enabled, "the Over/Under were left calibrated: identity kept"
    text = model.describe()
    assert text.startswith("Calibrazione applicata: sì") and "1X2: log loss -" in text and "GG/NG" in text and "partite di verifica" in text
    # the temperature softens the sharpened 1X2 (inverse temperature near 1 / 2.2) and Platt undoes the GG slope
    assert model.markets[Market.MATCH_ODDS].pooled.inv_temperature == pytest.approx(1 / 2.2, abs=0.08)
    assert model.markets[Market.BTTS].pooled.slope == pytest.approx(1 / 1.8, abs=0.08)
    # out of sample on brand-new matches from the same truth: better log loss than the distorted input
    fresh = overconfident(truth_frame(4000, seed=99, start=date(2026, 8, 1)))
    before = bt.metrics(fresh, by=None, markets=["1X2", "GG/NG"]).set_index("market")["log_loss"]
    after = bt.metrics(model.apply_frame(fresh), by=None, markets=["1X2", "GG/NG"]).set_index("market")["log_loss"]
    assert (after < before - 0.01).all()
    # the evaluation helper reports the same verdict on the later split (model fitted on the earlier part only)
    early_model = cal.fit_recalibration(shown, refit_full=False)
    report = cal.evaluate_recalibration(shown, early_model).set_index("market")
    for market in ("1X2", "GG/NG"):
        assert report.loc[market, "gain"] == pytest.approx(early_model.markets[Market(market)].gain, rel=1e-6)
        assert bool(report.loc[market, "enabled"]) and report.loc[market, "brier_after"] < report.loc[market, "brier_before"]
    assert report.loc["OU2.5", "gain"] == pytest.approx(0.0, abs=1e-12)


def test_platt_recovers_a_known_distortion() -> None:
    rng = np.random.default_rng(5)
    n = 30_000
    p_true = rng.uniform(0.15, 0.85, n)
    y = (rng.uniform(size=n) < p_true).astype(int)
    # shown = sigmoid((logit(true) - 0.3) / 0.6): the inverse map is logit(true) = 0.3 + 0.6 * logit(shown)
    shown = expit((logit(p_true) - 0.3) / 0.6)
    frame = pd.DataFrame({"date": pd.Timestamp(2026, 1, 1), "family": "Serie C", "p_GG": shown, "p_NG": 1 - shown, "gg": y})
    x, outcome = cal._design(frame, bt.market_spec("GG/NG"), "p_")
    params = cal._fit_params(x, outcome, bt.market_spec("GG/NG"), 0.0, cal.BinaryParams())
    assert params.intercept == pytest.approx(0.3, abs=0.05) and params.slope == pytest.approx(0.6, abs=0.03)


def test_l2_shrinks_a_small_sample_towards_the_identity() -> None:
    small = overconfident(truth_frame(80, seed=4))
    spec = bt.market_spec("1X2")
    x, y = cal._design(small, spec, "p_")
    free = cal._fit_params(x, y, spec, 0.0, cal.MultinomialParams())
    shrunk = cal._fit_params(x, y, spec, 200.0, cal.MultinomialParams())
    distance = lambda p: float(np.abs(p.vector() - np.array([1.0, 0.0, 0.0])).sum())  # noqa: E731
    assert distance(shrunk) < 0.25 * distance(free)
    assert cal._fit_params(x[:0], y[:0], spec, 5.0, cal.MultinomialParams()).is_identity


def test_families_get_their_own_parameters_only_with_enough_matches() -> None:
    serie_c = overconfident(truth_frame(4000, seed=6, families=("Serie C",)), power=2.5)
    serie_d = truth_frame(3000, seed=7, families=("Serie D",))
    serie_b = overconfident(truth_frame(150, seed=8, families=("Serie B",)), power=2.5)
    frame = pd.concat([serie_c, serie_d, serie_b], ignore_index=True).sort_values("date", kind="stable").reset_index(drop=True)
    frame["match_id"] = [f"F{i}" for i in range(len(frame))]
    model = cal.fit_recalibration(frame, markets=["1X2"], min_n=200)
    calibration = model.markets[Market.MATCH_ODDS]
    assert calibration.enabled and set(calibration.by_family) == {"Serie C", "Serie D"}, "Serie B (150 matches) uses the pooled parameters"
    assert calibration.by_family["Serie C"].inv_temperature < calibration.by_family["Serie D"].inv_temperature
    assert calibration.by_family["Serie D"].inv_temperature == pytest.approx(1.0, abs=0.12), "the calibrated family stays close to the identity"
    probs = {"1": 0.6, "X": 0.25, "2": 0.15}
    c_adjusted = model.apply({"1X2": probs}, family="Serie C")["1X2"]
    b_adjusted = model.apply({"1X2": probs}, family="Serie B")["1X2"]
    pooled = model.apply({"1X2": probs})["1X2"]
    assert c_adjusted["1"] < probs["1"] and b_adjusted == pooled and sum(c_adjusted.values()) == pytest.approx(1.0)


def test_small_evaluation_split_is_never_trusted() -> None:
    frame = overconfident(truth_frame(250, seed=12))
    model = cal.fit_recalibration(frame, min_eval=100)
    assert not model.enabled and all("partite di verifica" in c.reason for c in model.markets.values())
    one_day = truth_frame(300, seed=13).assign(date=pd.Timestamp(2026, 1, 1))
    flat = cal.fit_recalibration(one_day)
    assert not flat.enabled and all(c.reason.startswith("dati insufficienti") for c in flat.markets.values())
    assert cal.fit_recalibration(one_day.iloc[0:0]).markets == {}
    assert cal.RecalibrationModel.identity().describe().startswith("Calibrazione applicata: no")
    with pytest.raises(ValueError):
        cal.fit_recalibration(frame, l2=-1.0)


# ============================================================================ apply
def test_apply_keeps_markets_coherent_and_summing_to_one(calibrated: pd.DataFrame) -> None:
    shifted = calibrated.copy()
    gg = expit(1.2 + logit(np.clip(shifted["p_GG"].to_numpy(), 1e-6, 1 - 1e-6)))  # GG grossly under-predicted
    shifted = shifted.assign(p_GG=gg, p_NG=1 - gg)
    shifted = shifted.assign(p_O25=1 - shifted["p_U25"])
    model = cal.fit_recalibration(shifted)
    adjusted = model.apply_frame(shifted)
    for spec in bt.MARKET_SPECS.values():
        assert np.allclose(adjusted[list(spec.columns("p_"))].sum(axis=1), 1.0)
    assert (adjusted["p_O35"] <= adjusted["p_O25"] + 1e-12).all() and (adjusted["p_O25"] <= adjusted["p_O15"] + 1e-12).all()
    assert (adjusted["p_GG"] <= adjusted["p_O15"] + 1e-12).all()
    # the mapping form keeps its keys (Market members or strings) and selection order; unknown markets pass through
    single = {Market.OVER_UNDER_15: {"Over 1.5": 0.7, "Under 1.5": 0.3}, "GG/NG": {"GG": 0.69, "NG": 0.31}, "corner": {"x": 1.0}}
    out = model.apply(single, family="Serie C")
    assert list(out) == [Market.OVER_UNDER_15, "GG/NG", "corner"] and out["corner"] == {"x": 1.0}
    assert out["GG/NG"]["GG"] <= out[Market.OVER_UNDER_15]["Over 1.5"] + 1e-12 and sum(out["GG/NG"].values()) == pytest.approx(1.0)
    assert single["GG/NG"]["GG"] == 0.69, "the input is never modified"
    # switched off, the same model changes nothing but keeps its verdicts
    off = model.disabled()
    assert not off.enabled and off.markets[Market.BTTS].gain == model.markets[Market.BTTS].gain
    assert off.apply(single) == single


def test_apply_prediction_returns_a_corrected_copy(calibrated: pd.DataFrame) -> None:
    model = cal.fit_recalibration(overconfident(calibrated))
    matrix = dixon_coles_pmf(1.6, 0.9, -0.05)
    probabilities = markets_from_matrix(matrix)
    prediction = MatchPrediction(
        fixture=Fixture("FS-1", date(2026, 11, 15), 14, "A", "Lecco", "Trento"),
        home_rate=1.6,
        away_rate=0.9,
        rho=-0.05,
        matrix=matrix,
        probabilities=probabilities,
        fair_odds={m: {s: 1 / p for s, p in sels.items()} for m, sels in probabilities.items()},
    )
    original = matrix.copy()
    corrected = model.apply_prediction(prediction)
    assert corrected is not prediction and corrected.matrix is not prediction.matrix
    assert np.array_equal(prediction.matrix, original), "original matrix untouched"
    assert (corrected.home_rate, corrected.away_rate, corrected.rho) == (1.6, 0.9, -0.05), "the goal rates stay the model's"
    assert prediction.probabilities[Market.MATCH_ODDS] == markets_from_matrix(matrix)[Market.MATCH_ODDS], "original untouched"
    assert corrected.probabilities[Market.MATCH_ODDS]["1"] != prediction.probabilities[Market.MATCH_ODDS]["1"]
    for market, sels in corrected.probabilities.items():
        assert sum(sels.values()) == pytest.approx(1.0)
        for selection, p in sels.items():
            assert corrected.fair_odds[market][selection] == pytest.approx(1 / p)
    assert cal.RecalibrationModel.identity().apply_prediction(prediction) is prediction


def test_recalibrated_prediction_heatmap_cells_add_up_to_the_shown_markets(calibrated: pd.DataFrame) -> None:
    """Regression: with recalibration on, the heatmap (score matrix) and the most likely score must agree with the markets shown.

    Before the fix the matrix was the raw one: its Over 2.5 cells summed to the original model's probability
    while the panel showed the recalibrated P(Over 2.5) next to it.
    """
    distorted = overconfident(calibrated)
    # also distort Over 2.5 so that the correction touches a market built from the cells off the diagonal
    distorted["p_O25"] = expit(0.4 + 1.6 * logit(distorted["p_O25"].clip(1e-6, 1 - 1e-6)))
    distorted["p_U25"] = 1.0 - distorted["p_O25"]
    model = cal.fit_recalibration(distorted)
    assert {Market.MATCH_ODDS, Market.BTTS, Market.OVER_UNDER_25} <= set(model.enabled_markets)
    for lam, mu, rho in ((1.6, 0.9, -0.05), (0.7, 0.6, 0.08), (2.4, 1.8, -0.1)):
        matrix = dixon_coles_pmf(lam, mu, rho)
        probabilities = markets_from_matrix(matrix)
        prediction = MatchPrediction(
            fixture=Fixture("FS-2", date(2026, 11, 15), 14, "A", "Giana Erminio", "AlbinoLeffe"),
            home_rate=lam,
            away_rate=mu,
            rho=rho,
            matrix=matrix,
            probabilities=probabilities,
            fair_odds={m: {s: 1 / p for s, p in sels.items()} for m, sels in probabilities.items()},
        )
        corrected = model.apply_prediction(prediction)
        assert corrected.probabilities[Market.OVER_UNDER_25]["Over 2.5"] != pytest.approx(probabilities[Market.OVER_UNDER_25]["Over 2.5"], abs=1e-4)
        cells = markets_from_matrix(corrected.matrix)
        for market, sels in corrected.probabilities.items():
            for selection, p in sels.items():
                assert cells[market][selection] == pytest.approx(p, abs=1e-8), (market, selection)
        assert corrected.matrix.sum() == pytest.approx(1.0) and (corrected.matrix >= 0).all()
        home, away, p_top = corrected.most_likely_score()
        assert p_top == pytest.approx(float(corrected.matrix.max())) and corrected.matrix[home, away] == p_top


def test_rake_score_matrix_keeps_the_shape_and_leaves_free_markets_alone() -> None:
    matrix = dixon_coles_pmf(1.4, 1.1, -0.06)
    raw = markets_from_matrix(matrix)
    # targets equal to the matrix's own markets: nothing to do
    np.testing.assert_allclose(cal.rake_score_matrix(matrix, raw), matrix, atol=1e-12)
    # only 1X2 given (keyed by its value): it is met, the scores keep their order inside each part
    target = {"1X2": {"1": 0.40, "X": 0.32, "2": 0.28}, "corner": {"x": 1.0}}
    raked = cal.rake_score_matrix(matrix, target)
    got = markets_from_matrix(raked)[Market.MATCH_ODDS]
    assert got == pytest.approx(target["1X2"], abs=1e-9)
    diagonal = np.diag(matrix)
    assert np.allclose(np.diag(raked), diagonal * 0.32 / diagonal.sum()), "a part is rescaled as a whole: its shape is the model's"
    assert raked is not matrix and np.array_equal(matrix, dixon_coles_pmf(1.4, 1.1, -0.06)), "input untouched"
    with pytest.raises(ValueError):
        cal.rake_score_matrix(np.ones((2, 3)) / 6, raw)


def test_serialisation_round_trip_and_summary(calibrated: pd.DataFrame) -> None:
    model = cal.fit_recalibration(overconfident(calibrated))
    payload = json.loads(json.dumps(model.to_dict()))
    again = cal.RecalibrationModel.from_dict(payload)
    assert again.markets == model.markets and again.split_date == model.split_date and again.min_gain == model.min_gain
    probs = {"1X2": {"1": 0.5, "X": 0.3, "2": 0.2}}
    assert again.apply(probs, family="Serie D") == model.apply(probs, family="Serie D")
    summary = model.summary_frame()
    assert list(summary["market"]) == ["1X2", "GG/NG", "OU1.5", "OU2.5", "OU3.5"] and summary["enabled"].sum() == 2
    assert "inv_temperature=" in summary.loc[0, "parameters"]
    with pytest.raises(ValueError):
        cal.RecalibrationModel.from_dict({**payload, "schema_version": 99})


def test_time_split_cuts_on_whole_days() -> None:
    frame = truth_frame(1000, seed=3)
    earlier, later, cut = cal.time_split(frame, test_fraction=0.3)
    assert cut is not None and earlier["date"].max() < pd.Timestamp(cut) <= later["date"].min()
    assert 0.2 < len(later) / len(frame) < 0.4 and len(earlier) + len(later) == len(frame)
    explicit_early, explicit_late, explicit_cut = cal.time_split(frame, split="2025-08-10")
    assert explicit_cut == date(2025, 8, 10) and (explicit_late["date"] >= pd.Timestamp(2025, 8, 10)).all()
    assert cal.time_split(frame, split=date(2025, 8, 10))[2] == date(2025, 8, 10)
    with pytest.raises(ValueError):
        cal.time_split(frame, test_fraction=1.5)
    assert cal.time_split(frame.iloc[0:0])[2] is None


# ============================================================================ tuning
def test_tune_hyperparameters_ranks_a_small_grid() -> None:
    league = make_league({"A": 8, "DA": 8}, seed=21, current_rounds=5)
    steps: list[float] = []
    result = cal.tune_hyperparameters(
        league.matches,
        grid={"xi": [0.0, 0.0018], "prior": ["auto", None]},
        min_train_matches=60,
        progress=lambda fraction, _message: steps.append(fraction),
    )
    table = result.table
    assert len(table) == 4 and set(table["prior"]) == {"auto", "off"} and (table["n"] == 5 * 8).all()
    assert table["objective"].is_monotonic_increasing and result.best == {"xi": float(table["xi"].iloc[0]), "prior": str(table["prior"].iloc[0])}
    assert result.production is not None and math.isfinite(result.improvement) and result.improvement >= 0
    assert steps == [0.25, 0.5, 0.75, 1.0]
    assert "Ottimizzazione walk-forward: migliore" in result.describe()
    again = cal.TuningResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert again.best == result.best and again.production == result.production and len(again.table) == 4
    config = result.config_for(league.matches.iloc[:50], season="2026/27")
    assert config.xi == result.best["xi"] and config.method == "SLSQP"
    with pytest.raises(ValueError):
        cal.tune_hyperparameters(league.matches, grid={"xi": [-1.0], "prior": ["auto"]})


def test_rolling_recalibration_only_learns_from_earlier_matchdays() -> None:
    shown = overconfident(truth_frame(6000, seed=31))
    days = pd.Timestamp(2025, 9, 7) + pd.to_timedelta(7 * (np.arange(len(shown)) * 12 // len(shown)), unit="D")  # 12 weekly matchdays
    shown = shown.assign(date=days, matchday=days.strftime("%Y-%m-%d"))
    rolled = cal.rolling_recalibration(shown, min_history=1500, markets=["1X2", "GG/NG"])
    steps = rolled.steps
    assert len(steps) == shown["matchday"].nunique() and steps["n_history"].is_monotonic_increasing
    early = steps.loc[steps["n_history"] < 1500, "matchday"]
    untouched = shown["matchday"].isin(set(early))
    pd.testing.assert_frame_equal(rolled.frame.loc[untouched], shown.loc[untouched])
    assert rolled.n_corrected_matchdays > 0 and set(steps["enabled"]) - {""} <= {"1X2, GG/NG", "1X2", "GG/NG"}
    late = ~untouched
    before = bt.metrics(shown.loc[late], by=None, markets=["1X2"]).iloc[0]["log_loss"]
    after = bt.metrics(rolled.frame.loc[late], by=None, markets=["1X2"]).iloc[0]["log_loss"]
    assert after < before, "the out-of-sample correction helps on the later matchdays"
    # no leakage: rewriting the results of the last matchday never changes the earlier corrections
    last = shown["matchday"].max()
    altered = shown.copy()
    altered.loc[altered["matchday"] == last, "res_1X2"] = "2"
    again = cal.rolling_recalibration(altered, min_history=1500, markets=["1X2", "GG/NG"])
    keep = shown["matchday"] != last
    pd.testing.assert_frame_equal(again.frame.loc[keep, list(bt.PROB_COLUMNS)], rolled.frame.loc[keep, list(bt.PROB_COLUMNS)])
    assert cal.rolling_recalibration(shown.iloc[0:0]).steps.empty
