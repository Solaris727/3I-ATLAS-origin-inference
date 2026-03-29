# 3I-ATLAS Bayesian Inference Framework

[![GitHub release](https://img.shields.io/github/v/release/Solaris727/3I-ATLAS-origin-inference)](https://github.com/Solaris727/3I-ATLAS-origin-inference/releases/latest)
[![Zenodo](https://zenodo.org/badge/DOIEnter your DOI.svg)](https://doi.org/your-doi)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

**Two-stage hierarchical Bayesian framework** inferring stellar spectral class and protoplanetary disk chemistry of interstellar comet 3I/ATLAS (C/2025 N1) from JWST/NIRSpec measurements.

## Quick Summary

**Stage 1** (volatiles only): `p(S | D_vol)` → CO₂/H₂O = 7.6±0.3, CO/H₂O = 1.65±0.09  
**Stage 2** (prospective): `p(S | D_vol, D_iso)` → adds D/H = 0.95±0.06%, ¹²C/¹³C = 141–191  

**Key features:**
- Fisher Information Matrix identifiability gate (N_eff analysis)
- Two-layer GCR processing model (Γ_R retrievability)
- D/H fractionation + Luo (2024) GCE forward models
- MCMC/HMC-NUTS sampling + SVGP emulation
- HDF5 posterior data release planned

## Files
| File | Description |
|------|-------------|
| `3IATLAS_Inference_Framework_v1_0.md` | Complete technical specification |
| `zenodo.json` | Auto-generated Zenodo metadata |

## Usage
See `3IATLAS_Inference_Framework_v1_0.md` for:
- Full forward model equations
- Prior specifications  
- Sampling strategy
- Sensitivity analysis suite


## Disclosures
**AI Assistance:** Developed with Perplexity AI assistance. Author independently verified all:
- Observational data and citations
- Physical forward models (chemistry, GCR, fractionation, GCE)  
- Prior specifications and likelihood construction
- Bayesian methodology and diagnostics

AI handled document structure and prose only.

**License:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
