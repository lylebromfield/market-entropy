# Market Entropy

A topological data analysis (TDA) application for detecting market regime changes using persistent homology.

## Overview

This tool analyzes the structural complexity of financial markets by computing topological entropy from correlation networks and time-delay embeddings. It identifies periods when market structure breaks down, signaling potential crashes or regime shifts.

## Mathematical Foundation

### Correlation Topology

The application constructs distance matrices from rolling correlation windows of the Magnificent 7 stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA):

1. Compute pairwise correlations over a lookback window
2. Convert correlations to distances: d = √(2(1 - ρ))
3. Apply Vietoris-Rips persistence to extract topological features (H₀, H₁, H₂)
4. Calculate persistence entropy as a complexity measure

Low entropy indicates stocks moving in lockstep (危 danger). High entropy indicates diverse, independent behavior (healthy market).

### Temporal Topology

The Takens embedding reconstructs phase space dynamics from the S&P 500 price series:

1. Extract log returns from price data
2. Construct delay embeddings: X(t) = [x(t), x(t-τ), x(t-2τ)]
3. Compute persistence diagrams from embedded point clouds
4. Measure topological entropy of the attractor geometry

This captures how predictable vs. chaotic the price trajectory becomes over time.

### Combined Signal

Both entropy measures are z-score normalized and averaged. The signal is smoothed using exponential weighting. A historical percentile threshold (default: 5th percentile) defines critical regime territory.

## Status Interpretation

- **Healthy**: Entropy above threshold. Market structure is diverse and resilient.
- **Caution**: Entropy near threshold. Structure beginning to weaken.
- **Euphoria**: Entropy below threshold with rising prices. Dangerous correlation buildup during rally.
- **Critical Risk**: Entropy below threshold with falling prices. Structure collapse during decline.

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

Market data is fetched from Yahoo Finance starting from 2000. The analysis window is user-configurable (2000-present) with adjustable parameters for lookback window, smoothing, and threshold sensitivity.

## Visualization

### Signal Tab
Time series chart showing entropy signal, critical threshold, and S&P 500 price overlay.

### Inspector Tab
- **Correlation Map**: MDS projection showing distance relationships between stocks at selected date
- **Takens Attractor**: 3D phase space reconstruction showing temporal dynamics

## References

- Gidea, M., & Katz, Y. (2018). Topological data analysis of financial time series: Landscapes of crashes.
- Takens, F. (1981). Detecting strange attractors in turbulence.
- Edelsbrunner, H., & Harer, J. (2008). Persistent homology: A survey.
