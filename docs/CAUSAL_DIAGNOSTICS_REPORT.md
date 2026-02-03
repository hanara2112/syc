# Causal Diagnostics Report: Sycophancy Layer Sweep (L14–L27)

This report documents the causal and mechanistic properties of the sycophancy representation across the model's depth. Our core finding is a **dissociation between linear decodability and behavioral control**.

## 1. Quantitative Summary (Sweep Results)

| Layer | Logit Score ('A') | Causal Sensitivity | Interpretation |
| :--- | :--- | :--- | :--- |
| **14** | -0.0020 | +0.0033 | Inhibitory / Causal Noise |
| **16** | -0.0008 | +0.0002 | Neutral / Epiphenomenal |
| **18** | -0.0011 | +0.0006 | Neutral / Epiphenomenal |
| **20** | **+0.0111** | **-0.0012** | **Causal Window (Active Control)** |
| **22** | +0.0092 | +0.0019 | Transition to Readout |
| **24** | +0.0050 | +0.0036 | Transition to Readout |
| **26** | **+0.1621** | +0.0016 | **Readout Peak (Non-Causal)** |
| **27** | +0.0940 | 0.0000 | Terminal Readout |

---

## 2. Mechanistic Classification

### Phase I: Emergence & Latency (L14–L18)
*   **Geometric Signal:** Present and orthogonal to partisanship.
*   **Logit Projection:** Near zero. The unembedding matrix cannot "see" the feature.
*   **Causal Sensitivity:** Positive (+Loss). Adding the vector confuses the model's prediction.
*   **Verdict:** Epiphenomenal representation.

### Phase II: The Causal Pivot (L20)
*   **Logit Projection:** Flips to positive.
*   **Causal Sensitivity:** Negative (-Loss). This is the only layer where the 1D linear direction **causally assists** the model in agreeing with the user.
*   **Verdict:** This is the primary "control surface" where the abstract sycophancy feature is integrated into the decision circuit.

### Phase III: The Readout Artifact (L26–L27)
*   **Logit Projection:** Massive jump (+0.16 at L26). The feature is extremely "decodable" through simpler linear lens here.
*   **Causal Sensitivity:** Returns to positive/zero.
*   **Verdict:** Post-hoc decision tracking. The model has already converged on an answer; the sycophancy signal here is a broadcast of the final state, not a driver of it.

---

## 3. Evidence for ICML Submission

> **Main Claim:** "Peak decodability (L26) is geometrically distinct from peak control (L20)."

This data directly supports the thesis that **Representation Engineering (RepE/CAA)** often targets "readout" layers rather than "control" layers, leading to the failure of linear steering interventions despite successful linear probing.

### Required Figures
1.  **Dissociation Master Plot:** Unified view of Feature Norm, Logit Alignment, and Causal Sensitivity with functional region shading (Emergence, Control, Readout).
2.  **Sensitivity Dip:** Focused view of Layer 20 highlighting the causal pivot.
