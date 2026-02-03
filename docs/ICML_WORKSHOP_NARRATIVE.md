# Representation–Control Dissociation

**Title:** Linear Decodability Does Not Imply Causal Control: The Case of Sycophancy

---

## 1. The Core Scientific Problem

A central assumption in **Representation Engineering (RepE)** and **Mechanistic Interpretability** is that identifying a clean, linearly decodable feature in the residual stream provides a direct "handle" for causal behavioral control.

We demonstrate that this assumption is fundamentally flawed. Using **Sycophancy** as a case study, we show that a feature can be geometrically isolated and highly decodable through a linear lens, yet remain **causally inert** in the very layers where its representation is strongest.

---

## 2. Methodology: Factorial Isolation

We constructed a 2×2 factorial dataset that isolates sycophancy (the intent to agree) from political partisanship (underlying identity).

* **Geometric Discovery:** We establish that sycophancy is represented by a single linear direction ($v_{syc}$) that is orthogonal to political identity ($\cos < 0.1$ in mid-layers).
* **Decodability Peak:** This signal norm increases monotonically with depth, becoming extremely decodable at the unembedding layer (L26+).

---

## 3. The Discovery: Representation–Control Dissociation

By sweeping **Logit Projection** (Decodability) against **Causal Sensitivity** (Control), we discover a "Lag" in the mechanistic lifecycle of the feature:

1. **Phase I: Emergence (L14–L18):** The feature is present but "hidden" from decision circuits. Intervening here increases loss.
2. **Phase II: The Causal Window (L20):** A narrow mechanistic bottleneck where the feature is integrated into the model's decision circuit. This is the **true control surface**.
3. **Phase III: The Readout (L26+):** The feature norm and decodability peak. However, the decision is already "locked in," and steering here has negligible behavioral impact.

### Key Evidence:

* **Causal Gain:** Steering at Layer 20 produces **3x more logit drift** per unit of intervention than steering at the "Peak" Layer 26.
* **The Dissociation Lag:** Decodability peaks late, but control peaks early.

---

## 4. Impact & Implications

Our work suggests that many current "Sycophancy Detection" and "Steering" methods are suboptimal because they target **Readout Artifacts** in late layers rather than **Control Pivots** in middle layers.

**The Narrative Contribution:**
We provide a quantitative framework for distinguishing between **Control Representations** (which drive behavior) and **Readout Representations** (which merely track it). This distinction is critical for the safety and alignment of large-scale models where late-layer steering often leads to "over-steering" or feature-suppression without behavioral change.

---

## 5. Visual Proof

![Dissociation Master Plot](/Users/aryamanbahl/.gemini/antigravity/brain/5dfc0ffc-aa14-41a8-ac91-4dd59710794c/dissociation_master_plot.png)
*Figure 1: The unified master plot showing the divergence between Feature Norm (Readout) and Causal Sensitivity (Control).*
