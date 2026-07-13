# Crystalline Cohomology for ETFs

Uses p-adic cohomology to extract arithmetic invariants from discrete price grids. de Rham-Witt complexes capture microstructure effects invisible to real-valued analysis. Torsion information measures microstructure noise. The per‑ETF score is the torsion invariant.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- p-adic valuation of discretised price differences
- de Rham-Witt complex construction
- Torsion invariant = sum of valuations + variance
- Score = torsion (higher = more microstructure)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-crystalline-cohomology-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High torsion → more microstructure noise → potential alpha from microstructure effects.
- Low torsion → smoother price dynamics.

## Requirements

See `requirements.txt`.
