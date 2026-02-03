# Geometric Analysis Results: Sycophancy vs. Partisanship

## 1. Signal Strength & Layer Localization
*   **Peak Layer:** 30 (out of 32)
*   **Signal Strength (L2 Norm):** 2.9434
*   **Interpretation:** Sycophancy is a **late-layer feature**. The model likely computes "truth" or "factual" representations in earlier layers and then applies a "sycophancy adjustment" just before generating the output token. This is consistent with other "high-level" behavioral features in LLMs.

## 2. Orthogonality Check
*   **Cosine Similarity:** `-0.0817`
*   **Conclusion:** The Sycophancy Vector and the Partisanship Vector are **effectively orthogonal**.
*   **Implication:** The model treats "agreeing with the user" as a completely separate mechanism from "representing a political stance".
    *   You can be Sycophantic + Liberal
    *   You can be Sycophantic + Conservative
    *   You can be Non-Sycophantic + Liberal
    *   You can be Non-Sycophantic + Conservative
*   This suggests we can intervene on the Sycophancy vector (e.g., via activation steering) **without** accidentally shifting the model's political bias.

## 3. Geometry Visualization
The 2D projection reveals a clean **2x2 grid structure**:

| | **Liberal Persona** | **Conservative Persona** |
| :--- | :---: | :---: |
| **Model Agrees** | **Cluster A** (Top Right) | **Cluster C** (Top Left) |
| **Model Disagrees** | **Cluster B** (Bottom Right) | **Cluster D** (Bottom Left) |

*(Note: Left/Right positions depend on the sign of the extracted vector, but the separation is distinct).*

- **Y-Axis (Sycophancy):** Clearly separates "Agree" responses from "Disagree" responses.
- **X-Axis (Partisanship):** Clearly separates responses to Liberal prompts from responses to Conservative prompts.

## 4. Next Steps
Since the vectors are orthogonal and clean, we are ready to proceed to **Activation Steering (CAA)**.
*   **Goal:** Clamp the Sycophancy feature to "off" (or negative) to force the model to be truthful even when the user tries to bias it.
*   **Method:** Add a multiple of the inverse Sycophancy Vector (`-1 * v_syco`) to the residual stream at Layer 30 during inference.
