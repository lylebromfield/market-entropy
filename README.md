# Market Entropy

A topological data analysis dashboard for detecting market regime changes using persistent homology and Wasserstein drift.

## Overview

This tool analyzes the structural complexity of financial markets by computing topological entropy from correlation networks and time-delay embeddings, with an optional Wasserstein drift signal. It identifies periods when market structure breaks down, signaling potential crashes or regime shifts.

## Mathematical Foundation

### Correlation Topology (11 SPDR sectors)

The application constructs distance matrices from rolling correlation windows of the 11 SPDR sector ETFs (XLK, XLF, XLV, XLE, XLC, XLI, XLP, XLY, XLU, XLB, XLRE):

1. Compute pairwise correlations over a lookback window.
2. Convert correlations to distances: d = √(2(1 - ρ)).
3. Apply Vietoris-Rips persistence to extract topological features (H₀, H₁, H₂).
4. Calculate persistence entropy as a complexity measure.

Low entropy indicates sectors moving in lockstep (higher fragility). High entropy indicates diverse, independent behavior (healthier structure).

### Temporal Topology

The Takens embedding reconstructs phase space dynamics from the S&P 500 price series:

1. Extract log returns from price data.
2. Construct delay embeddings: X(t) = [x(t), x(t-τ), x(t-2τ)].
3. Compute persistence diagrams from embedded point clouds.
4. Measure topological entropy of the attractor geometry.

This captures how predictable vs. chaotic the price trajectory becomes over time.

### Signal Selection

You can choose which signal drives alarms, validation, and optimization:

- **TDA Entropy**: Correlation-based topology (default fallback)
- **Shannon Entropy**: Return-distribution entropy (visual/optional)
- **Wasserstein Drift**: Structural speed via giotto-tda (primary default)

Signals are z-score normalized and exponentially smoothed. Historical percentile thresholds define elevated/critical territory.

## Status Interpretation

- **Healthy**: Signal outside alarm band; structure resilient.
- **Caution**: Signal near threshold; structure weakening.
- **Elevated/Critical**: Signal breaches threshold with adverse price context (definitions adjust per chosen signal).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Access the dashboard at https://market-entropy.streamlit.app/

## Data

Market data is fetched from Yahoo Finance starting from 2000. The analysis window is user-configurable (2000–present) with adjustable parameters for lookback window, smoothing, and threshold sensitivity.

## Visualization

### Signal Tab
Time series chart showing the selected primary signal, critical threshold, optional overlays (TDA/Shannon/Wasserstein), and S&P 500 price.

### Inspector Tab
- **Correlation Map**: MDS projection showing distance relationships between sectors at the selected date.
- **Takens Attractor**: 3D phase space reconstruction showing temporal dynamics.

### Validation & Optimization
- Validation backtests the chosen primary signal against drawdowns/forward returns.
- Optimization searches parameter grids (grid search or walk-forward). Wasserstein runs invert drift internally so high drift still maps to risk.

## References

- Gidea, M., & Katz, Y. (2018). Topological data analysis of financial time series: Landscapes of crashes.
- Takens, F. (1981). Detecting strange attractors in turbulence.
- Edelsbrunner, H., & Harer, J. (2008). Persistent homology: A survey.
