# Bayesian Inference of the Stellar and Disk Environment of 3I/ATLAS from Volatile and Isotopic Constraints

**Version:** 1.0
**Date:** March 28, 2026

## Table of Contents

1. Introduction
2. Observational Data
   - 2.1 Stage 1 Data Vector D_vol
   - 2.2 Stage 2 Data Vector D_iso
   - 2.3 Caveats and Future Observable Candidates
3. Scientific Objectives
4. Conceptual Model and Scope
   - 4.1 Extended-Source Corrections
   - 4.2 Retrievability
   - 4.3 Information Budget
   - 4.4 Identifiability Gate
5. Model Parametrisation
6. Ejection Model
7. Forward Model
   - 7.1 Two-Layer Surface Composition
   - 7.2 Sublimation Physics
   - 7.3 Stellar-Disk Chemistry Correlations
   - 7.4 Volatile Ratio Observable Vector
   - 7.5 D/H Forward Model
   - 7.6 ¹²C/¹³C Forward Model
   - 7.7 Outer Disk and Dead Zone Fractionation
   - 7.8 Stage 2 Implementation
8. Priors
   - 8.1 Spectral Class Prior π(S)
   - 8.2 Formation Chemistry Priors π_form(φ | S)
   - 8.3 Disk Temperature Structure Prior
   - 8.4 Processing Priors π_evol(η)
   - 8.5 Sublimation Priors π_sub
   - 8.6 Ejection Hyperpriors π(β)
   - 8.7 Kinematic Prior and Ejection Epoch Sampling
   - 8.8 Stage 2 Nuisance Priors
9. Likelihood and Total Uncertainty
   - 9.1 Γ_R Importance Reweighting
10. Emulator and Computational Strategy
    - 10.1 Architecture
    - 10.2 Design and Validation
    - 10.3 Sampler
    - 10.4 Methodological Notes
11. Full Posterior
12. Inference Outputs and Reporting Policy
13. Posterior Data Release (HDF5)
14. Validation, Diagnostics, and Sensitivity Tests
    - 14.1 Mandatory Validations
    - 14.2 Sensitivity Suite
    - 14.3 Failure Modes to Report
15. Discussion
16. Conclusions
17. References

## 1. Introduction

The interstellar object 3I/ATLAS (C/2025 N1) provides a unique opportunity to constrain the formation environment of a body originating outside the Solar System. JWST observations have revealed extreme volatile enrichment — CO₂/H₂O = 7.6 ± 0.3, CO/H₂O = 1.65 ± 0.09 — and highly anomalous isotopic ratios: ¹²C/¹³C = 141–191 and D/H = (0.95 ± 0.06)%, differing dramatically from Solar System comets [1, 2]. These measurements, interpreted through protoplanetary disk chemistry models, isotopic fractionation theory, and Galactic chemical evolution, enable Bayesian inference of the stellar spectral type and disk conditions in 3I/ATLAS's natal system.

This document presents a two-stage Bayesian framework. Stage 1 uses JWST/NIRSpec volatile ratios to constrain stellar spectral class S and disk chemistry parameters φ, with all priors fixed before the isotopic measurements were published (January 2026). Stage 2 incorporates the March 2026 D/H and ¹²C/¹³C measurements as likelihood updates. The gap between stages constitutes a prospective validation: Stage 1 predicted cold, metal-poor formation before the decisive isotopic data existed.

Protoplanetary disk chemistry models show strong radial gradients in volatile abundances and C/O ratios [3, 4, 5, 6], systematic dependencies on stellar spectral type [7, 8, 9], and complex isotopic fractionation mechanisms including chemo-dynamical coupling and dead zone effects [10, 11, 12]. These developments are incorporated as priors and interpretive context. The primary inference architecture is a hierarchical Bayesian model with priors from disk chemistry, sublimation physics, GCR processing, and ejection statistics; posterior sampling via MCMC with convergence diagnostics; and nested-sampling evidence estimates and simulation-based calibration as cross-checks.

## 2. Observational Data

**Notation:** Throughout this document, `log` denotes the natural logarithm (ln). All σ values are natural-log uncertainties derived via the delta method. Base-10 logarithms are never used.

### 2.1 Stage 1 Data Vector D_vol

**Volatile ratio values:** Earlier preliminary JWST reductions reported CO₂/H₂O ≈ 8.0 ± 1.0. The final JWST/NIRSpec measurement (Cordiner et al. 2025, ApJL 991, L43) gives CO₂/H₂O = 7.6 ± 0.3 and CO/H₂O = 1.65 ± 0.09. The preliminary value is superseded and not used anywhere in this framework.

<table>
  <thead>
    <tr><th>Observable</th><th>Value</th><th>σ_ln</th><th>Derivation</th><th>Source</th></tr>
  </thead>
  <tbody>
    <tr><td>log(CO₂/H₂O)</td><td>log(7.6)</td><td>0.039</td><td>0.3/7.6</td><td>Cordiner et al. 2025 [1]</td></tr>
    <tr><td>log(CO/H₂O)</td><td>log(1.65)</td><td>0.054</td><td>0.09/1.65</td><td>Cordiner et al. 2025 [1]</td></tr>
    <tr><td>r_h at observation</td><td>3.32 AU</td><td>fixed</td><td>—</td><td>Ephemeris</td></tr>
  </tbody>
</table>

These two observables constitute D_vol. They are the only data used in Stage 1 inference and in all prior specification.

**Extended H₂O source caveat:** At r_h = 3.32 AU, observed Q_H₂O is dominated by sublimation from icy grains entrained in the coma rather than direct nucleus outgassing. Multiple independent measurements confirm this: Cordiner et al. (2025) report H₂O Q-curves showing no clear asymptote [1]; Lisse et al. (2025) find >99% of SPHEREx continuum flux from coma dust, with Q_CO₂ = 9.4 × 10²⁶ s⁻¹ and Q_H₂O < 1.5 × 10²⁶ s⁻¹ at 3σ [13, 14]; Belyakov et al. (2026) confirm H₂O at MIRI distances is coma-grain dominated [15]. The ratio CO₂/H₂O in D_vol therefore mixes nucleus-sourced CO₂ with coma-grain-sourced H₂O. A nuisance parameter f_{ext,H₂O} — the fraction of observed H₂O from extended coma sources at 3.32 AU — is incorporated into the forward model (§7.2). Prior: Beta(5, 2), reflecting likely >50% extended-source contribution. The correction is applied inside the forward model as:

    log(CO₂/H₂O)_model = log(CO₂/H₂O)_nucleus + log(1 − f_{ext,H₂O})^{−1}
    log(CO/H₂O)_model  = log(CO/H₂O)_nucleus  + log(1 − f_{ext,H₂O})^{−1}

The correction is applied once, inside the likelihood, not to the data vector D_vol — applying it to the data before the likelihood would double-count it. Note that the ratio CO/CO₂ = 1.65/7.6 = 0.217 is exactly f_{ext,H₂O}-invariant, as the shared H₂O correction cancels. This ratio is the cleanest nucleus-composition constraint in Stage 1.

### 2.2 Stage 2 Data Vector D_iso

Published after January 2026. Not used in prior specification. Enters inference only through Stage 2 likelihoods.

**Decision criteria for Stage 2 inclusion:** (i) a forward model predicting the observable from θ exists; (ii) the observable constrains a direction not already constrained by D_vol; (iii) the measurement is nucleus-dominated with no unresolved extended-source contamination.

**Confirmed Stage 2 entries:**

**D/H = (0.95 ± 0.06)%** from JWST/NIRSpec (Cordiner et al. 2026, arXiv:2603.06911, in review *Nature*) [2]. HDO and H₂O measured from the same NIRSpec spectrum; no extended-source problem. Forward model: §7.5. Constrains log T_form and fractionation baseline independently of D_vol.

**¹²C/¹³C (CO₂) = 141–191** from JWST/NIRSpec (Cordiner et al. 2026) [2]. Treated as N(166, 25²) — see note below. Forward model: §7.6. Constrains [Fe/H] and t★ via GCE track independently of D_vol.

**Note on ¹²C/¹³C Gaussian approximation:** The paper reports a range 141–191 without specifying whether this is a 68% credible interval, 95% interval, or measurement range. This framework treats it as ±1σ (half-range = 25). This is an explicit modelling assumption. If 141–191 is a 95% interval, σ = 25 overstates uncertainty by a factor of 2. σ_iso_model (§8.8) absorbs additional uncertainty. Sensitivity to this is tested via: (a) σ = 12.5 run assuming 95% CI interpretation, and (b) the Opitom et al. (2026) asymmetric CN measurement as an independent cross-check (§14.2).

**Conditional Stage 2 entry — excluded:**

CH₄/H₂O = 13.7% at 2.20 AU and 27% at 2.54 AU (Belyakov et al. 2026) [15]. Condition failed. Belyakov et al. confirm H₂O production is dominated by extended coma grain sublimation at MIRI distances. CH₄/H₂O does not represent nucleus ice composition and cannot enter the Stage 2 likelihood. Functions as a posterior predictive check (PPC) only (§12).

**Stage 2 PPC only (not in likelihood):**

¹²C/¹³C = 147⁺⁸⁷₋₄₀ from VLT/UVES CN (Opitom et al. 2026, arXiv:2603.07187) [16]: same physical quantity as the Cordiner CO₂ measurement but via CN, which is more susceptible to coma photodissociation fractionation. Cross-check only.

¹⁴N/¹⁵N = 343⁺⁴⁵⁴₋₁₂₄ from VLT/UVES (Opitom et al. 2026) [16]: requires nitrogen GCE forward model not present in this framework. Large asymmetric uncertainties. Deferred.

**Supplementary diagnostics (inform priors and PPCs; not in likelihood):**

H₂O/OH UV (Swift; Xing et al. 2025) [17]; SPHEREx pre-perihelion (Lisse et al. 2025) [13, 14]; Santana-Ros et al. 2025 A&A 693 A151 (spin 16.16 ± 0.01 h) [18]; Serra-Ricart et al. 2025 (P_rot = 15.5 ± 0.7 h) [19]; BHTOM Collaboration 2026 (P_rot = 15.98 ± 0.08 h) [20]; Hartman et al. 2026 (n = 5.167 ± 0.095; β = 0.0552 ± 0.0032 mag deg⁻¹) [21]; Hinkle et al. 2025 (JCMT HCN/CO) [22]; Yaginuma et al. 2025 [23]; Roth et al. 2026 ApJL 999 L32 (CH₃OH/HCN extended) [24]; Hoogendam et al. 2026 (KCWI Ni/Fe, C₂, C₃) [25]; Zhao et al. 2026 (post-perihelion optical) [26]; Biver et al. 2026 (outflow velocity validates CO₂-dominated regime at 3.32 AU) [27].

Q_H₂O absolute (Tan et al. 2026, arXiv:2601.15443) [28]: f_active cancels from all production rate ratios. Absolute Q enters only the PPC for the Swift OH detection.

CO/CO₂ post-perihelion evolution (Lisse et al. 2026, arXiv:2601.06759) [29]: r_h-dependent ratio not modelled in the fixed-epoch forward model. Qualitative PPC on the two-layer structure of §7.1.

**Earliest spectroscopic baselines:**

Opitom et al. (2025, arXiv:2507.05226) [30] — VLT/MUSE integral-field spectroscopy two days after discovery at r_h = 4.47 AU; red coma spectral slope (18 ± 4)%/1000Å; no C₂, NH₂, CN, or [OI] detected.

Rahatgaonkar et al. (2025, arXiv:2508.18382) [31] — VLT X-shooter and UVES from r_h ≈ 4.4 to 2.85 AU; spectral slope S′ = 18.3 ± 0.9 %/1000Å, D-type taxonomy; CN emission detected; Ni II with Fe I undetected; steep heliocentric scalings Q(Ni) ∝ r_h^(−7.7 ± 1.0).

Paek et al. (2026, arXiv:2602.12930) [32] — 7DT pre-perihelion CN coma observations.

Hutsemekers et al. (2025, arXiv:2509.26053) [33] — Pre-perihelion Ni/Fe evolution; transition from extreme to normal values.

**Nucleus size — published estimates:**

Three independent estimates spanning a factor of ~3:
- Hui et al. (2026, arXiv:2601.21569) — HST PSF subtraction; R_n = 1.3 ± 0.2 km (p_V = 0.04); axis ratio ≥ 2:1 [34]
- Forbes & Butler (2026, RNAAS 10, 12) — NGA time-lag; R_n ~ 820–1050 m [35]
- Thoss, Loeb & Burkert (2026, arXiv:2603.15735) — CO₂-only NGA; R_n ~ 420 m [36]
- Spada et al. (2026, arXiv:2603.00782) — NGA systematics analysis; size sensitive to outgassing model; supports kilometre-scale nucleus [37]

The spread across NGA models (~420 m to ~1050 m) reflects genuine outgassing model degeneracy rather than measurement inconsistency. R_n is marginalised using a log-uniform prior over [0.3, 2.0] km. Adding R_n as a likelihood term is deferred until NGA outgassing model degeneracy is resolved.

### 2.3 Caveats and Future Observable Candidates

**CH₄/H₂O:** Presence of CH₄ at depth requires T_form cold enough for natal condensation (~31 K under low-pressure disk conditions). This is a condensation constraint on the natal ice, not a sublimation temperature constraint — CH₄ trapped in amorphous water ice or clathrate hydrates releases at 60–120 K depending on ice structure. Post-perihelion timing of CH₄ outgassing does not directly constrain T_form; what constrains T_form is that subsurface CH₄ exists at all. These are distinct physical claims and must not be conflated in interpretation.

**OCS:** Tentatively detected in the primary NIRSpec spectrum; OCS/H₂O ≈ 0.065 ± 0.015 (Cordiner et al. 2025) [1]. The framework treats s as a free parameter with a weakly informative prior. Adding OCS/H₂O to D_vol would provide a third volatile constraint sensitive to disk sulfur chemistry and metallicity. Deferred pending verification that the OCS spatial extraction aperture does not introduce extended-source bias.

## 3. Scientific Objectives

1. **Stage 1:** Infer p₁(S | D_vol) — the posterior over stellar spectral class S ∈ {O, B, A, F, G, K, M} given volatile ratios {log(CO₂/H₂O), log(CO/H₂O)}, using priors fixed by January 2026.

2. **Stage 2:** Compute p₂(S | D_vol, D_iso) — the Stage 1 posterior updated by isotopic likelihoods L_DH and L_iso from the March 2026 measurements.

3. Infer posterior distributions for formation chemistry parameters φ identifiably constrained by the data (per identifiability gate §4.4). Report prior-dominated parameters explicitly.

4. Demonstrate prospective validation: show that the Stage 1 posterior was consistent with the March 2026 isotopic measurements before those measurements were incorporated, and quantify the Stage 2 update.

5. Release the Stage 1 posterior ensemble in HDF5 format as a community data product enabling future thermophysical follow-up by independent groups.

## 4. Conceptual Model and Scope

**Stage 1:** Observed coma volatile composition is a mixture of pristine bulk and GCR-processed surface contributions. Can this mixture be inverted to recover formation chemistry, or does GCR processing dominate the signal?

**Stage 2:** Do the isotopic measurements — published after Stage 1 prior specification — sharpen the posterior in the direction Stage 1 predicted? The difference between p₁ and p₂ is precisely the information contributed by the March 2026 isotopic data.

Time-dependent GCR dose integration and thermophysical evolution are not modelled in this framework; both are deferred to future work by other groups, enabled by the HDF5 community data release.

### 4.1 Extended-Source Corrections

The extended-source H₂O contribution at r_h = 3.32 AU is confirmed by multiple independent lines of evidence (§2.1). The nuisance parameter f_{ext,H₂O} with prior Beta(5,2) is incorporated into the forward model. The correction applies simultaneously to both volatile ratios since the H₂O denominator is shared. The forward model predicts observed ratios from nucleus ice ratios via the expressions in §2.1. The correction is applied once inside the likelihood — not to the data vector — to avoid double-counting.

### 4.2 Retrievability

Processing geometry:
- d_proc: depth of GCR-processed crust (m)
- z_active: sublimation-active depth scale (m)
- δ ≡ ln(d_proc / z_active)

    w_proc(δ) = tanh(e^δ)
    Γ_R = 1 − w_proc(δ)  ∈  [0, 1]

At d_proc ≈ 15–20 m and z_active = 2 m, δ ≈ ln(10) and Γ_R ≈ 0 (processing-dominated).

Γ_R applies only to volatile-ratio observables in D_vol. Isotopic observables in D_iso are not materially degraded by GCR processing at the relevant doses — D/H in residual water ice is preserved to first order (Maggiolo et al. 2020 [38]; de Barros et al. 2014 [39]), and ¹²C/¹³C fractionation by GCR is estimated at ≲2–5% via the kinetic isotope effect (¹³C/¹²C)^½ ≈ 1.04. Stage 2 constraints on T_form and [Fe/H] therefore remain valid even in the fully processing-dominated regime (Γ_R ≈ 0), where Stage 1 is weakest.

<table>
  <thead>
    <tr><th>Γ_R</th><th>Regime</th><th>Stage 1 (volatile ratios)</th><th>Stage 2 (isotopics)</th></tr>
  </thead>
  <tbody>
    <tr><td>&gt; 0.5</td><td>Pristine-dominated</td><td>Constrains c, h, log T_form</td><td>Constrains T_form, [Fe/H], t★</td></tr>
    <tr><td>0.2–0.5</td><td>Mixed</td><td>Partially constrains</td><td>Constrains T_form, [Fe/H], t★</td></tr>
    <tr><td>&lt; 0.2</td><td>Processing-dominated</td><td>Prior-dominated</td><td>Still constrains T_form, [Fe/H], t★</td></tr>
  </tbody>
</table>

### 4.3 Information Budget

<table>
  <thead>
    <tr><th>Data stream</th><th>Γ_R &gt; 0.5</th><th>Γ_R &lt; 0.2</th></tr>
  </thead>
  <tbody>
    <tr><td>log(CO₂/H₂O) ∈ D_vol [Stage 1]</td><td>Constrains c, log T_form</td><td>Constrains ξ_GCR, d_proc</td></tr>
    <tr><td>log(CO/H₂O) ∈ D_vol [Stage 1]</td><td>Constrains h</td><td>Constrains ξ_GCR</td></tr>
    <tr><td>log(D/H) ∈ D_iso [Stage 2]</td><td>Constrains log T_form (conditional on A_D, B_D)</td><td>Γ_R-independent</td></tr>
    <tr><td>¹²C/¹³C ∈ D_iso [Stage 2]</td><td>Constrains [Fe/H], t★ via GCE</td><td>Γ_R-independent</td></tr>
    <tr><td>CH₄/H₂O (PPC only)</td><td>T_form ≲ 31 K condensation PPC</td><td>Γ_R-independent</td></tr>
    <tr><td>OCS/H₂O</td><td>Constrains s, [Fe/H] (deferred; §2.3)</td><td>Weak</td></tr>
    <tr><td>Nucleus R_n, AR</td><td>Constrains ζ</td><td>Unaffected</td></tr>
    <tr><td>Ejection ε_ej</td><td>Secondary context in Stage 1</td><td>Secondary context</td></tr>
  </tbody>
</table>

### 4.4 Identifiability Gate

**This gate runs before Stage 1 production sampling.**

D_vol contains two numbers. The formation chemistry subspace φ has five dimensions; with evolution nuisance η, that is seven. At most two linear combinations can be constrained by D_vol.

**Procedure:** Compute the Fisher Information Matrix at the MAP estimate for each spectral class:

    I_ij = −E[∂² ln L_vol / ∂θ_i ∂θ_j] |_{θ_MAP}

**Well-constrained direction condition:** λ_k · σ²_{prior,k} > 5, where λ_k is the Fisher information eigenvalue for eigendirection k, and σ²_{prior,k} is the prior variance projected onto that direction. This tests whether the likelihood contributes more than 5× the prior variance in that direction. N_eff is the count of directions satisfying this condition.

<table>
  <thead>
    <tr><th>N_eff</th><th>Action</th></tr>
  </thead>
  <tbody>
    <tr><td>≥ 2</td><td>Proceed; report all φ marginals, flagging unconstrained directions explicitly</td></tr>
    <tr><td>1</td><td>Report only the one constrained direction; all other φ marginals labeled prior-dominated</td></tr>
    <tr><td>0</td><td>Chemistry uninformative; class comparison via ejection model and kinematic priors only</td></tr>
  </tbody>
</table>

**Expected outcome for Stage 1:** N_eff = 1–2. c and a combination of h/ξ_GCR are expected to be the constrained directions. ζ, s, and log T_form are expected to be prior-dominated in Stage 1 alone. The spectral-class posterior p₁(S | D_vol) remains meaningful because it integrates over all of φ-space and the ejection model discriminates between classes even when chemistry is flat.

**Stage 2 identifiability:** D/H and ¹²C/¹³C add two constraint directions that are Γ_R-independent and approximately orthogonal to D_vol directions. N_eff for the full four-observable system is expected to be 3–4.

**Kinematic classification tension:** Guo et al. (2025) classify 3I as thin-disk (~20× more probable via Toomre + E–L_z + Bensby diagnostics) [40]. Taylor & Seligman (2025) independently find ~12% probability of parent-star [Fe/H] ≤ −0.4 [41]. Both point away from the thick-disk metallicity prior. The Stage 2 ¹²C/¹³C likelihood provides a data-side constraint on [Fe/H] independent of the kinematic prior; the thin-disk sensitivity run (§14.2) tests whether the isotopic data resolve or entrench this tension.

## 5. Model Parametrisation

**Discrete:** S ∈ {O, B, A, F, G, K, M}

**Formation chemistry φ:**

    φ = {log T_form, c = ln(f_CO₂/f_H₂O), h = ln(f_CO/f_CO₂),
         ζ = logit(f_ice:rock), s = ln(f_OCS/f_H₂O)}

**Evolution nuisance η:**

    η = {δ, ξ_GCR}

ξ_GCR ∈ [0, 1]: GCR CO→CO₂ conversion efficiency; sampled in logit space; bounded by laboratory radiolysis (Pilling et al. 2010) [42].

**Sublimation and systematics κ:**
- α_{s,i}: per-species sublimation coefficients; temperature-dependent (Kossacki et al. 1999) [43]
- σ_{vap,i}: per-species vapour pressure log-scale uncertainty (Fray & Schmitt 2009 [44]; CO caveat §8.5)
- f_mantle: dust-mantle suppression; Beta(2,2)
- Δ_{m,1}, Δ_{m,2}: per-species retrieval offsets for log(CO₂/H₂O) and log(CO/H₂O)
- σ_model: global forward model error scale

**Stage 2 nuisance parameters:**
- A_D: D/H fractionation log-baseline
- B_D: effective fractionation characteristic temperature (ΔE/k equivalent)
- σ_iso_model: GCE forward model uncertainty on ¹²C/¹³C prediction
- σ_cross: off-block covariance between D_vol and D_iso measurement blocks

**Kinematic parameters:**
- [Fe/H]: parent-star metallicity
- t★: stellar age (Gyr)
- u_ej ~ Uniform(0,1) → τ_ej = t★ · F⁻¹(u_ej): ejection epoch

**Ejection hyperparameters β:** {γ₀, γ₁, γ₂} (§6)

**Nucleus:** R_n, AR — both with log-uniform priors; marginalised, not in likelihood.

**Note on f_active:** Cancels exactly from all production rate ratios. Not a sampled parameter.

**OCS note:** GCR radiolysis experiments of OCS at cometary doses are sparse. The working assumption that OCS is less efficiently destroyed than CO by GCR irradiation is explicitly unverified. Leave-one-out OCS sensitivity test (§14.2) applies.

**Total dimensionality:** Stage 1 ~24–28; Stage 2 adds A_D, B_D, σ_iso_model, σ_cross (~27–32 total).

## 6. Ejection Model

    ε_ej(S, φ, β) = σ(ln[f_giant/(1−f_giant)] + γ₀ + γ₁·ζ + γ₂·I_CO₂(φ))

where I_CO₂(φ) = exp(−½·((log T_form − ln 70 K)/0.77)²) and σ(·) = logit⁻¹(·).

**f_giant(S, [Fe/H]):** Empirical giant planet occurrence rates from Fischer & Valenti (2005) [45], Cumming et al. (2008) [46], Bonfils et al. (2013) [47]. All three surveys are solar-neighbourhood, <5 Gyr, near-solar-metallicity populations.

**Extrapolation under thick-disk prior:** Population synthesis models (Mordasini et al. 2012 [48]; Schlecker et al. 2021 [49]) find thick-disk giant planet occurrence is 10–20× lower than thin-disk occurrence, driven by reduced solid material at low metallicity. Applied under the thick-disk prior, the survey-derived f_giant values **overestimate ε_ej** by a factor of ~10–20 — a class-uniform suppression, not a K/M-favouring one. The sensitivity run halving f_giant for [Fe/H] < −0.3 (§14.2) tests the spectral-class discrimination impact.

**γ parameters:**
- γ₀ ~ N(−1.0, 1.0²): baseline log-odds
- γ₁ ~ N(+0.5, 0.5²): disk mass scaling via ζ
- γ₂ ~ N(+0.5, 0.8²): formation-radius proximity via I_CO₂(φ)

**I_CO₂ and cold formation compound effect:** The ejection model peaks at T_form = 70 K and suppresses ε_ej at very cold temperatures. At the D/H-implied T_form ≈ 17–18 K, I_CO₂ ≈ 0.19 — an ~80% suppression relative to peak. The Stage 2 push toward cold T_form therefore simultaneously suppresses ε_ej. The three-tier posterior decomposition (§12) reveals the separate contributions of chemistry and ejection model.

**Hot-star classes under D_iso:** Under Stage 2, D/H constrains T_form toward cold values. For O/B/A classes, disk T_form anchors far exceed 30 K — achieving the D/H-required formation temperatures would require radii ≫100 AU, where disk surface density is too low for giant planet formation, further suppressing f_giant. The D/H and f_giant constraints on hot-star classes compound; O/B/A exclusion in Stage 2 has two simultaneous lines of attack.

## 7. Forward Model

### 7.0 Purpose

Stage 1 map: θ = (S, φ, η, κ, β) → μ_vol(θ) = {log Q_{CO₂/H₂O}, log Q_{CO/H₂O}}

Stage 2 adds: → μ_iso(θ) = {ln(D/H)_model, ¹²C/¹³C_model}

### 7.1 Two-Layer Surface Composition

    X_i^surf = Γ_R · X_i^bulk + (1−Γ_R) · X_i^proc

**Mole fraction mapping from φ to bulk ice compositions:**

Step 1 — Unnormalised fractions:

    F_CO₂ = exp(c),  F_CO = exp(c+h),  F_H₂O = 1,  F_OCS = exp(s)

Step 2 — Normalise:

    Z = F_CO₂ + F_CO + F_H₂O + F_OCS
    X_i^bulk_ice = F_i / Z

Step 3 — Ice:rock partitioning:

    f_ice = sigmoid(ζ),   X_i^bulk = f_ice · X_i^bulk_ice

**Processed layer endpoints:**

    X_CO₂^proc = X_CO₂^bulk + ξ_GCR · X_CO^bulk
    X_CO^proc  = (1−ξ_GCR) · X_CO^bulk
    X_H₂O^proc = X_H₂O^bulk · (1−f_decomp)

    Baseline model:   X_OCS^proc = X_OCS^bulk                          [OCS unchanged by GCR]
    Sensitivity run:  X_OCS^proc = X_OCS^bulk · exp(−ξ_GCR/2)         [exponential depletion; §14.2]

OCS processing is uncertain; the baseline assumes no modification because OCS is not a direct intermediate in the CO→CO₂ pathway (Maggiolo et al. 2020) [38].

**CH₄ PPC diagnostic:** CO-before-CH₄ ordering (Belyakov et al. 2026) [15] is a qualitative PPC: near-surface CO was converted to CO₂ by radiolysis; CH₄ (not in the CO→CO₂ chain) survived at depth and was released post-perihelion when the thermal wave penetrated below the processed layer.

### 7.2 Sublimation Physics

    Q_i = f_mantle · A_eff · X_i^surf · α_{s,i}(T_surf) · P^model_{vap,i}(T_surf) / √(2π m_i k_B T_surf)

f_active and A_eff cancel from all production rate ratios. Surface temperature: T_surf(r_h) ∝ (L★/L☉)^(1/4) · r_h^(−1/2) at r_h = 3.32 AU.

**Extended-source correction (applied inside forward model):** For both volatile ratios the predicted observable includes:

    μ_vol,i(θ) = log(Q_i/Q_H₂O)_nucleus + log(1 − f_{ext,H₂O})^{−1}

where the nucleus ratio is computed from sublimation physics as above.

**Outflow velocity validation:** The anomalously low gas velocity (~0.2–0.3 km s⁻¹ at 3.32 AU; Biver et al. 2026) [27] is consistent with CO₂-dominated outgassing, validating the CO₂-dominated regime at the primary datum epoch.

### 7.3 Stellar-Disk Chemistry Correlations

Disk chemistry exhibits systematic dependencies on stellar spectral type that inform the conditional structure of priors (§8.2, §8.3) without entering the likelihood. These correlations are incorporated as prior structure, not as likelihood adjustments.

**Herbig Ae/Be stars (A-F types):** Higher UV flux and warmer temperatures in the inner disk [7]. H₂O, C₂H₂, HCN, and CH₄ more abundant in hot inner regions than T Tauri disks [7]. Water snowline location difficult to trace with HCO⁺ [9].

**T Tauri stars (F-G-K types):** Moderate UV flux. Ice-phase C/O ≈ 0.2–0.3 between CO₂ and CH₄ snowlines [3, 4]. Water snowline at 2–5 AU depending on accretion rate [6, 50]. Shadowed midplane regions show 5–10× more ices [8]. Volatile enrichment peaks of 2–18× initial values at 2–11 AU [5].

**M dwarfs:** Lower UV flux and cooler temperatures [7]. Snowlines closer to star [51]. Longer disk lifetimes [45].

**Radial disk structure:** C/O ratios vary from 0 to 1.4 radially, with ice-phase C/O ≈ 0.2–0.3 between CO₂ and CH₄ snowlines [3, 4]. Radial drift and viscous gas accretion move snowlines inward by up to 40–60% [6]. Water snowlines shift from ≈10 AU to 4 AU over 490 kyr as disks cool [3, 4].

### 7.4 Volatile Ratio Observable Vector

    μ_vol(θ) = {log Q_{CO₂/H₂O}, log Q_{CO/H₂O}} + [Δ_{m,1}, Δ_{m,2}]ᵀ

Δ_m applies only to volatile ratios; it is exactly zero for isotopic observables, which predict absolute values with no instrumental reference ratio.

### 7.5 D/H Forward Model

The D/H ratio in cometary water ice is set by temperature-dependent gas-phase fractionation during ice formation. The dominant pathway involves exothermic ion-molecule exchange (H₃O⁺ + HD ⇌ H₂DO⁺ + H₂ and analogues). Because the reaction is exothermic (ΔE > 0), D/H increases with decreasing temperature.

**Functional form:**

    ln(D/H)_model = A_D + B_D / T_form

The plus sign is required: as T_form → 0, D/H → ∞; as T_form → ∞, D/H → exp(A_D). A minus sign produces the wrong asymptotic behaviour.

**Calibration check at fiducial prior values (A_D = −16.9, B_D = 210 K):**

<table>
  <thead>
    <tr><th>T_form</th><th>Predicted D/H</th><th>Comment</th></tr>
  </thead>
  <tbody>
    <tr><td>10 K</td><td>exp(−16.9 + 21.0) = exp(4.1) ≈ 6%</td><td>Extreme — prior width limits</td></tr>
    <tr><td>18 K</td><td>exp(−16.9 + 11.7) = exp(−5.2) ≈ 0.55%</td><td>Near 3I/ATLAS observed value</td></tr>
    <tr><td>25 K</td><td>exp(−16.9 + 8.4) = exp(−8.5) ≈ 0.02%</td><td>Solar system comet range</td></tr>
    <tr><td>30 K</td><td>exp(−16.9 + 7.0) = exp(−9.9) ≈ 0.005%</td><td>Below solar system comets</td></tr>
  </tbody>
</table>

With ln(D/H)_obs = ln(0.0095) = −4.657, the prior-centre-implied T_form ≈ 17.1 K. For hotter disk anchors, the required A_D shift is: ~+1.5σ at T_form = 25 K; ~+3.2σ at 50 K; ~+3.7σ at 70 K. The D/H likelihood is therefore strongly discriminating against FGK and hotter classes.

**Stage 2 likelihood:**

    L_DH = N(ln(D/H)_obs ; A_D + B_D/T_form, σ²_ln_DH + σ²_model_DH)

    ln(D/H)_obs = ln(0.0095) = −4.657
    σ_ln_DH = 0.063  [delta method: 0.0006/0.0095]
    σ_model_DH ~ HalfNormal(0, 0.1²)

**GCR stability:** D/H in residual water ice is preserved to first order under GCR irradiation (Maggiolo et al. 2020 [38]; de Barros et al. 2014 [39]): HDO and H₂O have similar radiolytic cross-sections. A 20% GCR correction sensitivity run (§14.2) tests the impact if this assumption fails.

**Ionisation rate degeneracy:** Cordiner et al. (2026) explicitly state that enhanced ζ_CR is required alongside cold temperature to explain D/H = 0.95% [2]. A_D absorbs ζ_CR through fractionation efficiency η. The T_form constraint from D/H is therefore conditional on both T_form and ζ_CR — more precisely "cold formation under enhanced ionisation." σ_{A_D} = 2.5 spans the physically plausible ζ_CR range.

**Non-monotonic D/H:** D/H may be non-monotonic at very large radii (>100–200 AU) due to radial transport of isotopically exchanged inner-disk material. The A_D prior breadth partially accommodates this. Albertsson et al. (2014) show 2D turbulent mixing produces shallower radial D/H gradients than laminar models; cometary D/H values of ~2.5–10 × 10⁻⁴ are achievable within 2–20 AU in mixing models [10]. Furuya et al. (2013) show turbulent mixing decreases water ice D/H from ~2 × 10⁻² to 10⁻⁴–10⁻² in 10⁶ years [12]. These effects are absorbed into A_D and B_D priors rather than modelled explicitly. The B_D sensitivity run tests directional robustness.

### 7.6 ¹²C/¹³C Forward Model

    ¹²C/¹³C_model = f_GCE([Fe/H], t★)

Implemented as bicubic interpolation of the Luo et al. (2024, A&A 686, A142) GCE grid [52]. Grid spans [Fe/H] ∈ [−1.5, +0.5] and lookback time ∈ [0, 13.5] Gyr.

**Stage 2 likelihood:**

    L_iso = N(¹²C/¹³C_obs ; f_GCE([Fe/H], t★), σ²_iso + σ²_iso_model)

    ¹²C/¹³C_obs = 166,  σ_iso = 25

**GCE interpretation:** Cordiner et al. (2026) note that thick-disk and Galactic bulge populations have below-average ¹²C/¹³C relative to the current local ISM, because early rapid star formation produced relatively few intermediate-mass AGB stars — the primary ¹³C factories [2]. The observed high ¹²C/¹³C = 141–191 is therefore NOT a thick-disk chemistry signature in the standard nucleosynthesis sense. It is consistent with formation either: (a) in the outer Galaxy at low metallicity, where the Galactocentric ¹²C/¹³C gradient (Milam et al. 2005 [53]; Yan et al. 2023 [54]) reaches 100–150+; or (b) in the very early Galaxy (~10–12 Gyr ago) before significant ¹³C enrichment accumulated from AGB cycling, which the Luo et al. (2024) GCE model interprets as ¹²C/¹³C ~ 130–200 at [Fe/H] ≈ −0.4 and t★ = 10 Gyr. This is an "early Galaxy / outer disk chemistry signature" consistent with (but not uniquely diagnostic of) thick-disk progenitor age.

**GCR stability:** Carbon isotopes have the same electronic structure; radiolytic cross-sections differ only by the kinetic isotope effect (¹³C/¹²C)^½ ≈ 1.04 — a 4% effect. Estimated cumulative preferential ¹³CO₂ loss over ~10 Gyr at Maggiolo et al. (2025) dose rates is ≲2–5%, within σ_iso = 25. σ_iso_model absorbs residual uncertainty.

**Non-GCR fractionation:** Isotope-selective photodissociation can modulate molecular ¹²C/¹³C in hydrocarbons and nitriles, though the effect on CO is typically small. CO₂ photodissociation fractionation at r_h > 3 AU is ≲5% (Hutsemekers et al. 2008) [55], within σ_iso_model.

**Scope boundary — in-disk fractionation:** The Luo et al. GCE grid provides the baseline ¹²C/¹³C set by galactic nucleosynthesis — the order-of-magnitude signal. In-disk isotopic chemistry (±10–20% effects; Woods & Willacy 2009 [56]; Aikawa et al. 2022 [57]) sits on top of this baseline and is absorbed into σ_iso_model and the Opitom et al. CN cross-check, not into the primary likelihood.

### 7.7 Outer Disk and Dead Zone Fractionation

Isotopic fractionation in protoplanetary disks exhibits complex spatial variations incorporated here as supplementary context for prior specification and posterior interpretation, not as replacements for the GCE-based likelihood.

**Outer disk fractionation (r > 5 AU):** D/H ratios increase with radius due to lower temperatures favouring deuterium enrichment [10, 12]. Turbulent mixing transports water from midplane to atmosphere where it is destroyed; atomic oxygen returns to reform water with lower D/H [12]. ¹²C/¹³C remains high (>100) in the outer disk where selective photodissociation and isotope exchange are less efficient [56, 57].

**Dead zone effects (2–5 AU):** Non-monotonic temperature profiles due to self-gravity heating lead to non-monotonic D/H enrichment, with a local D/H peak around 2 AU and a dip around 3.5 AU (Ali-Dib et al. 2015) [11]. The A_D prior breadth (σ = 2.5) accommodates this behaviour.

**Chemo-dynamical coupling:** 2D turbulent mixing produces shallower radial D/H gradients than laminar models [10]. Cometary D/H ratios (~2.5–10 × 10⁻⁴) are achievable within ~2–20 AU in mixing models versus ~2–6 AU in laminar models [10].

**Implication for prior structure:** These mechanisms motivate the wide A_D prior (σ = 2.5) and the B_D range (N(210, 40²)), which together encompass the diversity of chemo-dynamical fractionation regimes without reducing that diversity to a single in-disk parameterisation.

### 7.8 Stage 2 Implementation — Targeted MCMC (Primary) or IS Reweighting (Diagnostic)

In high-dimensional posteriors where Stage 1 is prior-dominated in a direction Stage 2 constrains sharply — specifically, Stage 1 is prior-dominated in log T_form while D/H is sharply peaked relative to that prior — importance resampling will almost certainly degenerate.

**Decision rule:** Compute σ² = Var_{p₁}(log L_DH + log L_iso) from a short Stage 1 pilot run (~1,000 samples). Both L_DH and L_iso must be included; computing only Var(log L_iso) understates the degeneracy by omitting the D/H term. If σ² < 2: IS reweighting is reliable. If σ² ≥ 2 (expected): go directly to targeted Stage 2 MCMC.

**Targeted Stage 2 MCMC (primary):** Joint likelihood L_vol · L_DH · L_iso with original Stage 1 priors. Initialise at Stage 1 MAP. Samples p₂(θ | D_vol, D_iso) directly. HDF5 stores Stage 1 and Stage 2 samples separately.

**IS reweighting (diagnostic only, when σ² < 2):**

    w_k = L_DH(θ_k) · L_iso(θ_k) / Σ_j [L_DH(θ_j) · L_iso(θ_j)]
    n_eff = (Σ_k w_k)² / Σ_k w_k²

**Prospective validation check:** Before Stage 2, evaluate L_DH(θ_k) and L_iso(θ_k) on Stage 1 samples. If n_eff ≥ 100, Stage 1 already covers Stage 2 — a quantitative demonstration that Stage 1 predicted the isotopic result. If n_eff << 100, Stage 1 did not predict Stage 2 in detail, which is also a reportable result quantifying how much the isotopics update the posterior beyond what the volatiles implied.

## 8. Priors

**Prior specification timeline:** All priors in §8.1–§8.7 were fixed using data available by January 2026 — volatile ratios (Cordiner et al. 2025) [1], kinematic classifications (Guo et al. 2025 [40]; Taylor & Seligman 2025 [41]; Hopkins et al. 2025 [58]) and physical sublimation/GCR models. No isotopic measurement (Cordiner et al. 2026 [2]; Opitom et al. 2026 [16]) informed any prior. Stage 2 nuisance priors (§8.8) are specified from physical models, not from March 2026 measurements.

### 8.1 Spectral Class Prior π(S)

IMF-weighted baseline using Chabrier (2003) [59] and Kroupa (2001) [60]. Sensitivity test: uniform. Brown dwarfs excluded — CO-dominated ice is incompatible with observed CO₂ enrichment even after processing correction. O/B/A classes retained as negative controls.

**Prior distribution (IMF-weighted):**
- P(S = O) = 0.001
- P(S = B) = 0.010
- P(S = A) = 0.030
- P(S = F) = 0.060
- P(S = G) = 0.130
- P(S = K) = 0.220
- P(S = M) = 0.549 (remainder)

These values are fixed by stellar demographics and are not adjusted based on observed CO₂/H₂O or CO/H₂O values. Any such adjustment would constitute double-use of the primary Stage 1 likelihood observable as a prior — a circularity error. The likelihood naturally down-weights classes inconsistent with the data.

### 8.2 Formation Chemistry Priors π_form(φ | S)

**Sampler dependence:** Class-level hyperpriors on μ_log T are identified only under joint sampling over S (Option B, §10.3). Under per-class chains (Option A), σ_hyp is not sampled — the temperature anchor is treated as fixed at the class-level mean. This distinction must be tracked when comparing results between sampler options.

<table>
  <thead>
    <tr><th>Class</th><th>Anchor μ_log T<sup>disk</sup></th><th>Physical basis</th></tr>
  </thead>
  <tbody>
    <tr><td>O–A</td><td>ln(100 K)</td><td>Hot, massive disks (Chiang &amp; Goldreich 1997 [61])</td></tr>
    <tr><td>F–G</td><td>ln(70 K)</td><td>Solar-type analogues</td></tr>
    <tr><td>K–M</td><td>ln(50 K)</td><td>Cooler, extended ice lines (Mulders et al. 2015 [51])</td></tr>
  </tbody>
</table>

These anchors are compared against the Stage 1 posterior on log T_form as a post-check: if Stage 1 posterior on log T_form is dominated by these anchors (N_eff = 0 for log T_form from the identifiability gate), this must be stated explicitly. Under Stage 2, the D/H likelihood provides an independent constraint on log T_form that resolves prior dominance in this direction.

c, h: class-dependent normal priors; σ_c = 0.6, σ_h = 0.7.

ζ: N(0, 1.5²) in logit space; expected to be prior-dominated in Stage 1.

s (OCS): metallicity-coupled: s ~ N(μ_s([Fe/H]), 0.8²), μ_s = s₀ + α_S · [Fe/H], α_S ~ N(1.5, 0.5²).

### 8.3 Disk Temperature Structure Prior

The disk midplane temperature at the reference radius (5 AU) follows a class-conditional log-normal:

    p(T_disk | S) = LogNormal(μ_S, σ_S²)

where μ_S and σ_S are set by stellar type:
- O/B/A: μ = log(180 K), σ = 0.25
- F: μ = log(160 K), σ = 0.25
- G: μ = log(140 K), σ = 0.30
- K: μ = log(120 K), σ = 0.30
- M: μ = log(90 K), σ = 0.35

**Radial temperature profile:** T(r) = T_disk · (r / 5 AU)^(−q) where q ~ Normal(0.5, 0.1) is the temperature gradient exponent, informed by disk structure models [3, 4, 6]. These values encode the physical expectation that hotter stars have warmer disks — they are physically motivated, not adjusted to favour any spectral class based on the observables. Snowline evolution over disk lifetime (Molyarova et al. 2021, 2024 [3, 4]; Piso et al. 2015 [6]) motivates the σ width: water snowlines can shift inward by 40–60% over disk evolution, placing the CO₂ snowline at 5–8 AU in evolved disks.

### 8.4 Processing Priors π_evol(η)

δ ~ N(ln 10, 1²): centred on d_proc ≈ 20 m with z_active = 2 m (Maggiolo et al. 2025) [62]. σ_δ = 1 spans approximately a factor of 3 in each direction, covering d_proc values from ~6 m to ~55 m.

ξ_GCR ~ logit⁻¹(N(0, 1)): bounded by laboratory radiolysis (Pilling et al. 2010) [42]. GCR processing primarily affects molecular abundances (CO→CO₂ conversion); isotopic ratios are preserved to first order; residual isotopic modification is absorbed into σ_iso_model.

### 8.5 Sublimation Priors π_sub

ln α_{s,i}: log-normal centred on Kossacki et al. (1999) [43]; σ = 0.2 per species.

σ_{vap,i}: HalfNormal with σ₀ = {0.3 (H₂O), 0.5 (CO₂), **1.5 (CO)**, 0.7 (OCS)}.

**CO vapour pressure caveat:** Grundy et al. (2023, arXiv:2309.05078) find the Fray & Schmitt (2009) [44] CO vapour pressure parameterisation overestimates P_vap(CO) by nearly an order of magnitude in the relevant temperature range [63]. σ_{vap,CO} = 1.5 (spanning ~±3.5 orders at 2σ) is set wide enough to encompass this systematic. A dedicated sensitivity run comparing parameterisations is included in §14.2; if CO/H₂O shifts by more than 1σ between parameterisations, the h and ξ_GCR posteriors must be reported as vapour-pressure-model-dependent.

f_mantle: Beta(2,2).

### 8.6 Ejection Hyperpriors π(β)

Priors centred on plausible values from Fischer & Valenti (2005) [45], Cumming et al. (2008) [46], Bonfils et al. (2013) [47]:
- γ₀ ~ N(−1.0, 1.0²)
- γ₁ ~ N(+0.5, 0.5²)
- γ₂ ~ N(+0.5, 0.8²)

### 8.7 Kinematic Prior and Ejection Epoch Sampling

**Prior specification date: January 2026.** These priors use kinematic data only — no isotopic measurement.

**Thick-disk conditional prior (baseline):**

    [Fe/H] ~ N(−0.40, 0.15²) truncated to [−0.7, −0.1]
    t★ ~ Uniform(6, 12 Gyr)

Source: Fuhrmann (2011) [64]; Recio-Blanco et al. (2014) [65].

**Kinematic tension:** Guo et al. (2025) find 3I is ~20× more kinematically probable as thin-disk [40]. Taylor & Seligman (2025) independently find ~12% probability of parent-star [Fe/H] ≤ −0.4 [41]. Both point away from the thick-disk [Fe/H] prior. The thick-disk prior is retained as baseline because the March 2026 isotopic measurements independently favour metal-poor, cold chemistry — but this is documented as a prior assumption, not a resolved conclusion. The Stage 2 ¹²C/¹³C likelihood provides data-side constraint on [Fe/H] independent of the kinematic prior.

**Thin-disk conditional prior (sensitivity run — §14.2):**

    [Fe/H] ~ N(−0.05, 0.20²) truncated to [−0.4, +0.4]
    t★ ~ Uniform(1, 8 Gyr)

Hopkins et al. (2025) ISO population model finds objects with 3I/ATLAS's velocity preferentially ejected from metal-poor stars under the thick-disk prior [58].

**Ejection epoch:**

    τ_ej = t★ · F⁻¹(u_ej),  u_ej ~ Uniform(0, 1)

F: log-normal CDF with μ_F = ln(0.5 Gyr), σ_F = 1.0. τ_ej stored in HDF5 as a derived quantity with the correct joint distribution under (t★, ε_ej).

### 8.8 Stage 2 Nuisance Priors

**A_D** (D/H fractionation log-baseline):

    A_D ~ N(−16.9, 2.5²)

Centred by calibration against solar system comets: Halley/Hale-Bopp D/H ~ 3 × 10⁻⁴ at T_form ~ 25 K gives ln(3 × 10⁻⁴) = A_D + B_D/25 → A_D = −8.11 − 8.80 = −16.9. σ = 2.5 covers ~factor of 150 in fractionation efficiency, spanning the ~2 order-of-magnitude variation Taquet et al. (2013) [66] show across physically plausible values of disk density, ζ_CR, and OPR at fixed T_form. A_D absorbs ζ_CR dependence, so the T_form constraint from D/H is conditional on both T_form and ζ_CR being in the high-fractionation regime. The fractionation energy constants ΔE/k for the leading ion-molecule channels cluster around 186–232 K (Ceccarelli et al. 2014 [67]; Taquet et al. 2013 [66]), setting the B_D prior centre.

**B_D** (effective fractionation characteristic temperature):

    B_D ~ N(210, 40²) K

σ = 40 K marginalises over multi-pathway competition, network model differences (Ceccarelli et al. vs. Taquet et al.), and OPR-dependent correction terms. B_D sensitivity run (§14.2, ±80 K) tests robustness.

**σ_iso_model** (GCE forward model uncertainty on ¹²C/¹³C):

    σ_iso_model ~ HalfNormal(0, 20²)

Absorbs three sources in quadrature: (i) ¹²C/¹³C Gaussian approximation uncertainty (~12 units if 141–191 is a 95% CI); (ii) GCE intrinsic scatter from radial migration and local ISM inhomogeneity (~10–20 units at [Fe/H] ≈ −0.4, t★ ~ 10 Gyr); (iii) AGB yield prescription uncertainty at 10 Gyr lookback (~15–20 units). Quadrature sum: ~25–30 units → HalfNormal(0, 20²) is conservative.

**σ_cross** (off-block covariance between D_vol and D_iso):

    σ_cross ~ HalfNormal(0, 0.01²)

## 9. Likelihood and Total Uncertainty

**Stage 1 likelihood:**

    L_vol(D_vol | θ) = N(D_vol ; μ_vol(θ), Σ_vol)

    Σ_vol = Σ_obs,vol + Σ_model + Σ_emul + Σ_sublim

where Σ_obs,vol = diag(0.039², 0.054²); Σ_model: σ_model ~ HalfNormal(0, 0.5²); Σ_emul: SVGP posterior predictive variance; Σ_sublim: propagated α_{s,i} and σ_{vap,i} uncertainty.

**Stage 2 likelihoods (analytic — no emulator):**

    L_DH(θ) = N(ln(D/H)_obs ; A_D + B_D/T_form, σ²_ln_DH + σ²_model_DH + Σ_frac)
    L_iso(θ) = N(¹²C/¹³C_obs ; f_GCE([Fe/H], t★), σ²_iso + σ²_iso_model)

CH₄/H₂O is not a Stage 2 likelihood term and functions as a PPC only.

### 9.1 Γ_R Importance Reweighting

Applied post-hoc to Stage 1 samples for the p_chem product:

    p̃₁(θ | D_vol, ν) ∝ p₁(θ | D_vol) · Γ_R(θ)^ν

For Stage 2:

    p̃₂(θ | D_vol, D_iso, ν) ∝ p₁(θ | D_vol) · Γ_R(θ)^ν · L_DH(θ) · L_iso(θ)

Sensitivity: ν ∈ {0.5, 1, 2}. Headline results use ν = 0.

## 10. Emulator and Computational Strategy

### 10.1 Architecture

Sparse Variational Gaussian Process (SVGP) with ARD Matérn-5/2 kernel (GPyTorch; Gardner et al. 2018 [68]; Titsias 2009 [69]). Training cost O(NM²). Emulates the Stage 1 volatile ratio block {log Q_{CO₂/H₂O}, log Q_{CO/H₂O}} — a 2D output. Stage 2 isotopic forward models are analytic and require no emulation.

### 10.2 Design and Validation

- Training set: N_train ~ 8,000–12,000 prior-weighted draws
- Inducing points: M ~ 400–600; ceiling M_max = 1,000
- **Acceptance criteria:**
  - (a) Posterior-weighted RMSE < observational σ (≈ 0.04–0.05) on prior-drawn held-out set
  - (b) 90% predictive intervals calibrated at 88–92% coverage
  - (c) Criterion (a) on 500 prior-tail samples
  - (d) Posterior-concentrated validation: after ~25% of Stage 1 production run, draw 300 approximate-posterior samples and evaluate exact forward model vs. emulator. RMSE must remain below observational uncertainty within the region containing ≥95% of posterior mass. If not: add 2,000 targeted evaluations in high-posterior-density region and retrain.
- **Fallback:** inflate σ_model by residual RMSE in quadrature; document inflation.
- Active learning: 1,000 exact evaluations in high-posterior-density region after initial burn-in.

### 10.3 Sampler

**Option A — ptemcee:** 8–10 temperature chains; power-law ladder β_k = (k/K)^5; target swap acceptance 20–40%; Gelman–Rubin R̂ < 1.01 and Geweke z-scores within ±2. Per-class chains combined via marginal likelihood weighting using thermodynamic integration ≥200 bootstrap replicates (Lartillot & Philippe 2006) [70], with dynesty (Speagle 2020) [71] as a cross-check for the top-2 candidates.

**Option B — NumPyro (joint HMC, preferred):** Joint HMC with discrete S marginalised analytically, using the NUTS implementation of Hoffman & Gelman (2014) [72]. Preferred if it converges stably, as it handles the full joint continuous parameter space without per-class chain combination.

**Stage 2 implementation:** Run the σ² pilot check on ~1,000 Stage 1 samples. If σ² < 2, IS reweighting is reliable. If σ² ≥ 2 (expected), go directly to targeted Stage 2 MCMC with joint likelihood L_vol · L_DH · L_iso.

**Simulation-based calibration:** Rank-uniformity diagnostics (Benavoli et al. 2021) [73] run on held-out synthetic data to validate sampler calibration before production.

### 10.4 Methodological Notes

**HMC with NUTS:** Czekala et al. (2019) demonstrate that HMC-NUTS via automatic differentiation significantly improves efficiency in high-dimensional correlated hierarchical models [74]. Shabram et al. (2020) confirm that HMC provides Gelman-Rubin statistics and ESS diagnostics to identify numerical bias and model pathologies [75]. The Stage 1 forward model must be differentiable for HMC-NUTS to apply, which is achievable in JAX/NumPyro with the two-layer sublimation model and SVGP.

**Adaptive Parallel Tempering:** Jenkins et al. (2025) demonstrate adaptive parallel tempering with Curvature-aware Thermodynamic Integration and Geometric-Bridge Stepping Stones matches or surpasses dynamic nested sampling while preserving posterior information [76]. This is the preferred algorithm for Stage 2 targeted MCMC when σ² ≥ 2.

**Flow Matching:** Gebhard et al. (2023) show that combining flow matching posterior estimation with neural importance sampling outperforms nested sampling in accuracy and simulation efficiency [77]. This is an optional alternative for Stage 2 given the high-dimensional joint posterior.

**Convergence criteria:** R̂ < 1.01 and ESS > 400 per parameter for all chains (Shabram et al. 2020 [75]; Sestovic et al. 2018 [78]).

## 11. Full Posterior

**Stage 1:**

    p₁(θ | D_vol) ∝ L_vol · π_form(φ|S) · π_evol(η) · π_sys(κ) · π(β) · π_sub · π(S)

**Stage 2:**

    p₂(θ | D_vol, D_iso) ∝ p₁(θ | D_vol) · L_DH(θ) · L_iso(θ)

<table>
  <thead>
    <tr><th>Group</th><th>Parameters</th><th>Count</th></tr>
  </thead>
  <tbody>
    <tr><td>Formation φ</td><td>log T_form, c, h, ζ, s</td><td>5</td></tr>
    <tr><td>Evolution η</td><td>δ, ξ_GCR</td><td>2</td></tr>
    <tr><td>Systematics κ</td><td>Δ_{m,1}, Δ_{m,2}, σ_model, f_{ext,H₂O}</td><td>4</td></tr>
    <tr><td>Ejection β</td><td>γ₀, γ₁, γ₂</td><td>3</td></tr>
    <tr><td>Sublimation</td><td>α_{s,i}, σ_{α,i}, σ_{vap,i}, f_mantle</td><td>7–9</td></tr>
    <tr><td>Kinematic</td><td>[Fe/H], t★, u_ej (→ τ_ej)</td><td>3</td></tr>
    <tr><td>Nucleus</td><td>R_n, AR (log-uniform prior)</td><td>2</td></tr>
    <tr><td>Stage 2 nuisance</td><td>A_D, B_D, σ_iso_model, σ_cross</td><td>4</td></tr>
    <tr><td><strong>Stage 1 total</strong></td><td></td><td><strong>~24–28</strong></td></tr>
    <tr><td><strong>Stage 2 total</strong></td><td></td><td><strong>~27–32</strong></td></tr>
  </tbody>
</table>

## 12. Inference Outputs and Reporting Policy

**Headline results:** p₁_full(S | D_vol) and p₂_full(S | D_vol, D_iso), both at ν = 0.

**Three-tier posterior products (reported under both stages):**

    p_form(S) ∝ ∫ L · π_form(φ|S) · π_evol(η) · π_sys(κ) · π(S) dφ dη dκ   [formation-only]
    p_full(S) ∝ p_form(S) × E[ε_ej | S, φ]                                   [headline]
    p_chem(S | Γ_R > 0.5) = p_full(S) restricted to Γ_R > 0.5 samples        [pristine regime]

p_chem threshold tested at Γ_R > {0.3, 0.5, 0.7}.

**Stage comparison reporting:**
- Report p₁_full and p₂_full side-by-side for each spectral class
- Report Stage 2 effective sample size n_eff from pilot IS diagnostic
- Report Stage 2/Stage 1 posterior ratio for the headline spectral class
- Report KL(p₂ || p₁) as information gain from isotopic data
- If p₁ and p₂ are substantially different: the isotopics resolved prior uncertainty. If similar: Stage 1 already captured the isotopic signal.

**Marginal posteriors reported under both stages:** log T_form, c, h, ζ, s, δ, ξ_GCR, Γ_R, ε_ej, [Fe/H], t★. Additional Stage 2 marginals: A_D, B_D. Report the joint (A_D, T_form) posterior as a 2D contour — the A_D/T_form degeneracy makes individual marginals less informative than their joint distribution.

**Posterior predictive checks:**
- Stage 1 quantitative: log(CO₂/H₂O), log(CO/H₂O)
- Stage 2 quantitative: log(D/H), ¹²C/¹³C
- Cross-checks: ¹²C/¹³C(CN) from Opitom et al. 2026 [16]; absolute Q_H₂O Swift OH detection (Tan et al. 2026) [28]; CO/CO₂ post-perihelion evolution (Lisse et al. 2026) [29]
- Qualitative: CH₄/H₂O ordering (CO detected before CH₄ — GCR two-layer model prediction)

## 13. Posterior Data Release (HDF5)

Files: `posterior_v1.0_{class}.h5`. Deterministic RNG seed, code commit SHA. Released as a standalone community data product enabling future thermophysical follow-up by independent groups, including time-dependent GCR dose integration using the Gronoff et al. (2020) H-BON10 model [79] and volatile loss modelling. The HDF5 includes sufficient metadata (deterministic RNG seeds, code commit SHA, emulator checkpoints) for full reproduction.

<table>
  <thead>
    <tr><th>Path</th><th>Contents</th></tr>
  </thead>
  <tbody>
    <tr><td>/stage1/posterior_samples/{φ,η,β,κ,S}</td><td>Full Stage 1 posterior samples</td></tr>
    <tr><td>/stage1/posterior_samples/t_star, u_ej</td><td>Stellar age and ejection timing</td></tr>
    <tr><td>/stage1/derived/tau_ej</td><td>Ejection epoch = t★·F⁻¹(u_ej)</td></tr>
    <tr><td>/stage1/derived/Gamma_R</td><td>Retrievability per sample</td></tr>
    <tr><td>/stage1/derived/epsilon_ej</td><td>Ejection efficiency per sample</td></tr>
    <tr><td>/stage1/derived/r_form</td><td>Inferred formation radius per sample</td></tr>
    <tr><td>/stage1/derived/Gamma_R_weights</td><td>Importance weights ν ∈ {0.5, 1, 2}</td></tr>
    <tr><td>/stage2/importance_weights</td><td>w_k = L_DH·L_iso / Σ L_DH·L_iso</td></tr>
    <tr><td>/stage2/effective_sample_size</td><td>n_eff of Stage 2 IS diagnostic</td></tr>
    <tr><td>/stage2/sigma_sq_pilot</td><td>Var_{p₁}(log L_DH + log L_iso) from pilot</td></tr>
    <tr><td>/stage2/derived/DH_model</td><td>Predicted D/H per Stage 1 sample</td></tr>
    <tr><td>/stage2/derived/C_iso_model</td><td>Predicted ¹²C/¹³C per Stage 1 sample</td></tr>
    <tr><td>/emulator/inducing_points</td><td>SVGP inducing point locations</td></tr>
    <tr><td>/emulator/hyperparams</td><td>Kernel hyperparameters</td></tr>
    <tr><td>/emulator/validation</td><td>Hold-out RMSE, calibration, σ_model inflation</td></tr>
    <tr><td>/observational_data/*</td><td>All reference metadata</td></tr>
    <tr><td>/importance_weights</td><td>Ẑ_S · π(S) / Z_total</td></tr>
  </tbody>
</table>

## 14. Validation, Diagnostics, and Sensitivity Tests

### 14.1 Mandatory Validations

- **Identifiability gate (§4.4):** FIM at MAP for each class; report N_eff and identifiable eigendirections before Stage 1 production sampling.
- **Prior predictive checks:** All four observables span plausible range; D/H prior generates values 0.02%–5% across T_form ∈ [10, 100] K; ¹²C/¹³C prior generates values ~50–250 across the [Fe/H] and t★ grid.
- **Post-Stage-1 anchor check:** Compare Stage 1 posterior on log T_form against class-level prior anchors (§8.2). If dominated by anchors (N_eff = 0 for log T_form), state explicitly.
- **Stage 1 PPCs:** log(CO₂/H₂O) and log(CO/H₂O) predicted distributions vs. observations.
- **Stage 2 PPCs:** log(D/H) and ¹²C/¹³C predicted distributions; cross-check ¹²C/¹³C(CN) vs. Opitom et al. 2026 [16].
- **Prospective validation check:** Evaluate L_DH(θ_k) and L_iso(θ_k) on Stage 1 samples before Stage 2 inference. If n_eff ≥ 100, Stage 1 predicted the isotopic result quantitatively.
- **σ² pilot check:** Report Var_{p₁}(log L_DH + log L_iso) alongside Stage 2 implementation path.
- **Retrievability validation:** Synthetic data with known Γ_R.
- **Emulator diagnostics:** Criteria (a)–(d) §10.2; σ_model inflation documented.
- **SBC calibration:** Rank-uniformity diagnostic on held-out synthetic data.

### 14.2 Sensitivity Suite (Minimum)

- IMF vs. uniform π(S)
- δ prior shifts: μ_δ ∈ {ln 5, ln 10, ln 15}
- β = (γ₀, γ₁, γ₂) priors at 2× variance
- Leave-one-out OCS: remove s from φ; recompute D_KL on log T_form
- Fix f_mantle = 1
- Gaussian vs. Student-t likelihood
- ν ∈ {0.5, 1, 2} for Γ_R^ν reweighting
- OCS sensitivity branch: X_OCS^proc = X_OCS^bulk · exp(−ξ_GCR/2) vs. baseline
- GCR profile amplitude ×0.5 and ×2.0
- [Fe/H] and t★ fixed at prior means
- Fix α_s = 1 (bare Hertz–Knudsen)
- γ₁ = 0 fixed
- f_giant halved for [Fe/H] < −0.3: tests thick-disk planet occurrence suppression (direction confirmed: 10–20× lower than thin-disk)
- Thin-disk [Fe/H] and t★ prior: [Fe/H] ~ N(−0.05, 0.20²) ∩ [−0.4, +0.4]; t★ ~ Uniform(1, 8 Gyr)
- B_D prior shifted ±80 K: tests robustness of Stage 2 T_form constraint
- A_D prior width: σ = 1.5 vs. 2.5
- D/H GCR 20% correction: verify T_form posterior shifts < 1σ
- Stage 2 without D/H: importance weights from L_iso only
- Stage 2 without ¹²C/¹³C: importance weights from L_DH only
- σ_iso_model doubled (HalfNormal(0, 40²)): tests ¹²C/¹³C constraint fragility
- CO vapour pressure: Grundy et al. (2023) [63] vs. Fray & Schmitt (2009) [44]; if CO/H₂O shifts > 1σ, report as vapour-pressure-model-dependent
- f_{ext,H₂O} = 0 vs. Beta(5,2) prior
- σ_cross = 0: tests block-diagonal Σ_obs assumption
- Opitom CN ¹²C/¹³C instead of Cordiner CO₂: replace 166 ± 25 with 147⁺⁸⁷₋₄₀
- ¹²C/¹³C 95% CI interpretation: use σ_iso = 12.5 instead of 25
- Radial gradient uncertainty: q ~ Normal(0.5, 0.2) vs. Normal(0.5, 0.1)
- I_CO₂ centre shift: replace ln(70 K) with ln(50 K); tests whether ejection model biases toward solar-type temperatures
- Disk chemistry model uncertainty: vary ice-phase C/O predictions by ±30%
- Stellar-disk correlation strength: vary p(T_disk | S) hyperparameters by ±20%

### 14.3 Failure Modes to Report

- Emulator miscoverage in prior tails
- Degenerate ejection/composition trade-offs
- Sensitivity of p₁_full to π(S) under low Γ_R
- **Kinematic prior dominance:** if p₁_full shifts substantially between thick-disk and thin-disk prior runs, Stage 1 is prior-dominated in the [Fe/H] direction
- **Stage 2 IS degeneracy (expected, not failure):** σ² = Var_{p₁}(log L_DH + log L_iso) expected to be large. Report σ² from pilot. If σ² ≥ 2 and targeted Stage 2 MCMC was used, report this as the primary implementation path.
- **A_D/B_D prior dominance:** if Stage 2 T_form posterior is insensitive to B_D, the D/H constraint is model-driven by A_D shape rather than fractionation energy
- **¹²C/¹³C tension with thin-disk prior:** if Stage 2 ¹²C/¹³C likelihood pulls [Fe/H] strongly away from the thin-disk prior centre, quantify the tension magnitude

## 15. Discussion

**Prospective validation and Stage 2 narrative:** The [Fe/H], t★, and log T_form prior centres were fixed by January 2026 using kinematic data (Guo et al. 2025 [40]; Hopkins et al. 2025 [58]; Taylor & Seligman 2025 [41]) and volatile-ratio measurements (Cordiner et al. 2025) [1]. The isotopic measurements of Cordiner et al. (2026) [2] and Opitom et al. (2026) [16], published March 2026, were not available at prior specification. The consistency of those measurements with the Stage 1 posterior-favoured parameter space — cold formation, metal-poor, old — constitutes a prospective validation. Stage 2 then formally incorporates those measurements. If Stage 2 sharpens in the direction Stage 1 pointed, this demonstrates Bayesian updating working as intended; each observation enters exactly once.

**What the D/H number implies:** D/H = 0.95% = 9.5 × 10⁻³ is ~30–60× enriched relative to solar system comparators (Hale-Bopp: ~3 × 10⁻⁴; 67P/C-G: ~5.3 × 10⁻⁴). At prior-centre values A_D = −16.9, B_D = 210 K, this implies T_form ≈ 17.1 K. For hotter disk anchors, the required A_D shift is 1.5σ at 25 K, 3.2σ at 50 K, 3.7σ at 70 K — strongly discriminating against FGK and hotter classes unless the fractionation environment was unusually efficient (high ζ_CR, dense molecular cloud). Cordiner et al. (2026) explicitly invoke enhanced ionisation to explain this value [2], which is absorbed into A_D.

**I_CO₂ interaction:** Cold formation (T_form ≈ 17–20 K) simultaneously suppresses ε_ej via I_CO₂ by ~80% relative to its peak at 70 K. The three-tier decomposition (p_form vs. p_full vs. p_chem) reveals whether the spectral class preference emerges from the formation chemistry likelihood or from the ejection model.

**Model-driven vs. prior-driven:** Stage 1 is model-driven in the formation chemistry direction and prior-driven in log T_form. Stage 2 is additionally model-driven in log T_form and [Fe/H] through the fractionation and GCE forward models. This is genuine progress — the framework moves from prior-dominated to model-dominated in two directions. However, model-driven is distinct from purely data-constrained: the T_form constraint from D/H is as strong as the fractionation model, and the [Fe/H] constraint from ¹²C/¹³C is as strong as the GCE grid. Both are physically motivated and well-cited, but they are model assumptions.

**¹²C/¹³C interpretation — early Galaxy, not thick disk:** High ¹²C/¹³C = 141–191 is not a thick-disk chemistry signature. Thick-disk populations have below-average ¹²C/¹³C due to rapid early star formation depleting intermediate-mass AGB ¹³C production [2]. The signal is consistent with formation before significant AGB cycling — either early Galaxy (~10–12 Gyr ago) or outer Galactic disk at low metallicity [52, 53, 54]. The Stage 2 thin-disk sensitivity run distinguishes between these scenarios.

**Kinematic tension resolution:** The three physically viable resolutions are: (A) thin-disk star with very early/outer disk formation environment — most consistent with all available data; (B) intermediate-metallicity origin where the GCE track also produces elevated ¹²C/¹³C at early times; (C) genuine posterior bimodality. The Stage 2 thin-disk sensitivity run tests which dominates.

**CH₄ as a formation temperature indicator:** Presence of CH₄ at depth requires T_form cold enough for natal condensation (~31 K). This is a condensation constraint on the natal ice, not a sublimation temperature of the current coma. CH₄ trapped in amorphous water ice or clathrate hydrates can be retained to 60–120 K before release. Post-perihelion timing of CH₄ outgassing says little about T_form directly.

**OCS as a future tracer:** The baseline assumption (OCS unchanged by GCR) is explicitly unverified. The leave-one-out OCS sensitivity run determines whether this affects the headline result. OCS/H₂O ≈ 0.065 ± 0.015 could constrain disk sulfur chemistry and metallicity if confirmed nucleus-dominated.

**Sublimation model uncertainty:** The largest modelling uncertainty internal to Stage 1 is the sublimation physics block. The CO vapour pressure issue — Fray & Schmitt (2009) [44] overestimates P_vap(CO) by nearly an order of magnitude in the relevant temperature range, as confirmed by Grundy et al. (2023) [63] — is the most significant single sublimation uncertainty. Because CO/H₂O is a primary observable in D_vol, this propagates directly into the h posterior and ξ_GCR. The dedicated sensitivity run (§14.2) determines whether the CO/H₂O posterior is vapour-pressure-model-dependent at the >1σ level.

**f_giant extrapolation:** Under the thick-disk prior, ε_ej is overestimated by ~10–20× uniformly across all spectral classes — a class-uniform suppression that reduces overall ejection efficiency without strongly favouring K/M over G.

**Disk chemistry context:** The literature on disk radial gradients [3, 4, 5, 6], stellar-type dependencies [7, 8, 9], and chemo-dynamical fractionation [10, 11, 12, 56, 57] supports the physical motivations for the prior structure in §8.2–8.3, provides posterior interpretation context for the inferred r_form, and motivates additional sensitivity runs (radial gradient uncertainty, I_CO₂ centre shift, disk chemistry model uncertainty). These references enrich the physical narrative around the posteriors; they do not replace the GCE-based likelihood or the kinematic prior.

## 16. Conclusions

This document presents a two-stage Bayesian framework for inferring the stellar and disk formation environment of interstellar object 3I/ATLAS from volatile and isotopic measurements. The framework is released as a community inference resource accompanying the HDF5 posterior data product.

1. **Stage 1 posterior:** p₁(S | D_vol) marginalises over full formation chemistry φ, GCR processing η, sublimation systematics κ, and ejection model β. The identifiability gate determines which parameter directions are data-constrained vs. prior-dominated before production sampling.

2. **Stage 2 update:** p₂(S | D_vol, D_iso) incorporates D/H = (0.95 ± 0.06)% and ¹²C/¹³C = 166 ± 25, both Γ_R-independent. The D/H measurement discriminates against FGK and hotter classes at high statistical significance. The ¹²C/¹³C measurement constrains [Fe/H] and t★ via the Luo et al. (2024) GCE grid [52], favouring metal-poor, early-Galaxy or outer-disk formation.

3. **Prospective validation:** The Stage 1 posterior was fixed before isotopic data existed. The degree to which Stage 1 predicted the Stage 2 posterior is quantified by n_eff from the IS pilot diagnostic and KL(p₂ || p₁).

4. **Kinematic tension:** The thin-disk kinematic classification and the early-Galaxy isotopic signature are not necessarily contradictory — reconciled by thin-disk membership with sufficiently early formation. The Stage 2 thin-disk sensitivity run quantifies whether the data resolve this tension.

5. **Community data product:** The Stage 1 posterior ensemble is released as a HDF5 data product, enabling future thermophysical follow-up including time-dependent GCR dose integration using the H-BON10 model [79] and volatile loss modelling.


## 17. References

[1] Cordiner, M. A., et al. 2025, ApJL, 991, L43

[2] Cordiner, M. A., et al. 2026, arXiv:2603.06911

[3] Molyarova, T., et al. 2021, ApJ, 910, 153

[4] Molyarova, T., et al. 2024, PASA, 42, e001

[5] Schneeberger, A., et al. 2023, A&A, 670, A78

[6] Piso, A.-M. A., et al. 2015, ApJ, 815, 109

[7] Agúndez, M., et al. 2018, A&A, 616, A19

[8] Notsu, S., et al. 2022, ApJ, 936, 188

[9] Leemker, M., et al. 2021, A&A, 646, A3

[10] Albertsson, T., et al. 2014, ApJ, 784, 39

[11] Ali-Dib, M., et al. 2015, A&A, 583, A58

[12] Furuya, K., et al. 2013, ApJ, 779, 11

[13] Lisse, C. M., et al. 2025, arXiv:2512.07318

[14] Lisse, C. M., et al. 2025, RNAAS, 9, 293

[15] Belyakov, M., et al. 2026, arXiv:2601.22034

[16] Opitom, C., et al. 2026, arXiv:2603.07187

[17] Xing, Z., et al. 2025, arXiv:2508.04675

[18] Santana-Ros, T., et al. 2025, A&A, 693, A151

[19] Serra-Ricart, M., et al. 2025, arXiv:2512.12819

[20] BHTOM Collaboration 2026, arXiv:2603.01383

[21] Hartman, J. D., et al. 2026, AJ, in press

[22] Hinkle, J. T., et al. 2025, arXiv:2512.02106

[23] Yaginuma, A., Taylor, A. G., & Seligman, D. Z. 2025, arXiv:2510.25945

[24] Roth, N. X., et al. 2026, ApJL, 999, L32

[25] Hoogendam, W. B., et al. 2026, arXiv:2601.16983

[26] Zhao, R., et al. 2026, arXiv:2603.07718

[27] Biver, N., et al. 2026, A&A, submitted

[28] Tan, H., et al. 2026, ApJL, in press

[29] Lisse, C. M., et al. 2026, arXiv:2601.06759

[30] Opitom, C., et al. 2025, arXiv:2507.05226

[31] Rahatgaonkar, R., et al. 2025, arXiv:2508.18382

[32] Paek, G. M., et al. 2026, arXiv:2602.12930

[33] Hutsemekers, D., et al. 2025, arXiv:2509.26053

[34] Hui, M.-T., et al. 2026, arXiv:2601.21569

[35] Forbes, J. C., & Butler, H. 2026, RNAAS, 10, 12

[36] Thoss, V., Loeb, A., & Burkert, A. 2026, arXiv:2603.15735

[37] Spada, F., Królikowska, M., & Dones, L. 2026, arXiv:2603.00782

[38] Maggiolo, R., et al. 2020, ApJ, 901, 136

[39] de Barros, A. L. F., et al. 2014, MNRAS, 438, 2026

[40] Guo, X., et al. 2025, arXiv:2509.07678

[41] Taylor, A. G., & Seligman, D. Z. 2025, ApJL, 990, L14

[42] Pilling, S., et al. 2010, A&A, 523, A77

[43] Kossacki, K. J., et al. 1999, Planet. Space Sci., 47, 1521

[44] Fray, N., & Schmitt, B. 2009, Planet. Space Sci., 57, 2053

[45] Fischer, D. A., & Valenti, J. 2005, ApJ, 622, 1102

[46] Cumming, A., et al. 2008, PASP, 120, 531

[47] Bonfils, X., et al. 2013, A&A, 549, A109

[48] Mordasini, C., et al. 2012, A&A, 541, A97

[49] Schlecker, M., et al. 2021, A&A, 656, A71

[50] Sasselov, D. D., & Lecar, M. 2000, ApJ, 528, 995

[51] Mulders, G. D., et al. 2015, ApJ, 807, 9

[52] Luo, Y.-J., et al. 2024, A&A, 686, A142

[53] Milam, S. N., et al. 2005, ApJ, 634, 1126

[54] Yan, Y. T., et al. 2023, A&A, 670, A109

[55] Hutsemekers, D., et al. 2008, A&A, 490, L31

[56] Woods, P. M., & Willacy, K. 2009, ApJ, 693, 1360

[57] Aikawa, Y., et al. 2022, arXiv:2212.14529

[58] Hopkins, M. J., et al. 2025, ApJL, 990, L30

[59] Chabrier, G. 2003, PASP, 115, 763

[60] Kroupa, P. 2001, MNRAS, 322, 231

[61] Chiang, E. I., & Goldreich, P. 1997, ApJ, 490, 368

[62] Maggiolo, R., et al. 2025, arXiv:2510.26308

[63] Grundy, W. M., et al. 2023, arXiv:2309.05078

[64] Fuhrmann, K. 2011, MNRAS, 414, 2893

[65] Recio-Blanco, A., et al. 2014, A&A, 567, A5

[66] Taquet, V., Ceccarelli, C., & Kahane, C. 2013, A&A, 549, A44

[67] Ceccarelli, C., et al. 2014, in Protostars and Planets VI, ed. H. Beuther et al. (Tucson: Univ. Arizona Press), 859

[68] Gardner, J., et al. 2018, arXiv:1809.11165

[69] Titsias, M. K. 2009, in Proc. 12th Int. Conf. AISTATS, PMLR 5, 567

[70] Lartillot, N., & Philippe, H. 2006, Syst. Biol., 55, 195

[71] Speagle, J. S. 2020, MNRAS, 493, 3132

[72] Hoffman, M. D., & Gelman, A. 2014, JMLR, 15, 1593

[73] Benavoli, A., Wyse, J., & White, A. 2021, arXiv:2109.13891

[74] Czekala, I., et al. 2019, ApJ, 883, 22

[75] Shabram, M., et al. 2020, AJ, 160, 16

[76] Jenkins, J. S., et al. 2025, arXiv:2509.24870

[77] Gebhard, T. D., et al. 2023, arXiv:2312.08295

[78] Sestovic, M., et al. 2018, A&A, 616, A76

[79] Gronoff, G., et al. 2020, ApJ, 890, 89

[80] Nomura, H., et al. 2022, in ASP Conf. Ser. 534, Protostars and Planets VII, ed. S. Inutsuka et al. (San Francisco: ASP), 1

[81] Bockelée-Morvan, D., et al. 2024, arXiv:2406.11526

[82] Owen, A. B., & Zhou, Y. 2000, JASA, 95, 135

[83] Xiao, Y., et al. 2024, arXiv:2406.01864

[84] Hourihane, S., et al. 2022, arXiv:2212.06276

[85] Mamajek, E. E. 2009, in AIP Conf. Proc. 1158, ed. T. Usuda et al. (Melville, NY: AIP), 3

[86] Bergin, E. A., et al. 2007, in Protostars and Planets V, ed. B. Reipurth, D. Jewitt, & K. Keil (Tucson: Univ. Arizona Press), 751

[87] Rasmussen, C. E., & Williams, C. K. I. 2006, Gaussian Processes for Machine Learning (MIT Press)

[88] Bockelée-Morvan, D., et al. 2000, A&A, 353, 1101

[89] Mumma, M. J., & Charnley, S. B. 2011, ARA&A, 49, 471

[90] Öberg, K. I., Murray-Clay, R., & Bergin, E. A. 2011, ApJL, 743, L16

[91] Kama, M., et al. 2019, ApJ, 885, 114

[92] Tobin, J. J., et al. 2023, Nature, 615, 227

[93] Cooke, R. J., Pettini, M., & Steidel, C. C. 2018, ApJ, 855, 102

[94] Bensby, T., Feltzing, S., & Oey, M. S. 2014, A&A, 562, A71

[95] Bensby, T., Feltzing, S., & Lundström, I. 2003, A&A, 410, 527

[96] Hutsemekers, D., et al. 2026, in prep.

[97] Kossacki, K. J., & Leliwa-Kopystynski, J. 2014, Icarus, 233, 101

[98] Kossacki, K. J., & Markiewicz, W. J. 2013, Icarus, 224, 172

[99] Bergner, J. B., et al. 2024, arXiv:2408.04538

[100] Manfroid, J., et al. 2009, A&A, 503, 613

[101] Law, C. J., et al. 2025, arXiv:2511.09628

[102] Keyte, L., et al. 2024, MNRAS, 528, 388

[103] Blum, J., et al. 2017, MNRAS, 469, S755

[104] Prialnik, D., et al. 2004, in Comets II, ed. M. C. Festou, H. U. Keller, & H. A. Weaver (Tucson: Univ. Arizona Press), 359

[105] Gundlach, B., Skorov, Yu. V., & Blum, J. 2011, Icarus, 213, 710

[106] Dong, J., et al. 2023, ApJS, 268, 73

[107] Ida, S., & Lin, D. N. C. 2004, ApJ, 604, 388

[108] Bowler, B. P. 2016, PASP, 128, 102001

[109] Heck, D. W. 2019, Br. J. Math. Stat. Psychol., 72, 316

[110] Wagenmakers, E.-J., et al. 2010, Cogn. Psychol., 60, 158

[111] Calderhead, B., & Girolami, M. 2009, Comput. Stat. Data Anal., 53, 4028

[112] Rossi, S., et al. 2021, arXiv:2003.03080
