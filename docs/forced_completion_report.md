# Forced Completion Done Right: Avoiding Confounds in Activation Extraction

## Executive Summary
Based on the CAA paper (Rimsky et al., 2023) and related work, this report synthesizes best practices for forced completion activation extraction. The key insight is that confounds can be managed through careful experimental design, not avoided entirely.

## The Core Problem Revisited
When we append " (A)" or " (B)" to a prompt and extract activations, we capture four distinct signals:
1.  **Letter identity**: "This activation came from token (A)"
2.  **Answer semantics**: "The conservative answer was chosen"
3.  **Behavioral signal**: "This was a sycophantic response"
4.  **Prompt encoding**: "The user is conservative"

We want to isolate **#3 (behavioral signal)**, but all four are present.

## How CAA (Rimsky et al.) Solves This

### Key Principle: Averaging Cancels Confounds
"By only varying the answer option between paired prompts and keeping the rest of the prompt constant, we isolate the internal representation most related to the target behavior while canceling out other confounding variables." — CAA Paper, Section 3

### The Math
For each contrast pair (same prompt, different completions):
$$ \Delta = \text{activation}(\text{prompt} + \text{positive}) - \text{activation}(\text{prompt} + \text{negative}) $$

When averaged over many pairs:
*   **Letter identity**: Cancels if A/B balanced (half positive=(A), half positive=(B)).
*   **Prompt encoding**: Cancels (same prompt in both).
*   **Answer semantics**: Partially cancels across varied questions.
*   **Behavioral signal**: **ACCUMULATES**.

## Implementation Details

### 1. Extraction Position: At the Answer Letter
**Critical**: Compute the difference within the activations *at the position of the answer letter*.
*   **Prompt**: "Question: ... (A) Yes (B) No"
*   **Positive completion**: "(A)"
*   **Negative completion**: "(B)"

### 2. A/B Balancing: Essential
Ensure within EACH label (sycophantic, non-sycophantic):
*   50% of examples have (A) as the answer.
*   50% of examples have (B) as the answer.

### 3. Layer-Wise Emergence
*   **Layers 0-9**: Letter clustering dominates (trivial).
*   **Layers 10-25**: Behavioral signal emerges (for 32-layer models).
*   **Layer 32**: Strong behavioral signal but may include output formatting.

**Recommendation**: Focus on layers 10-24 for Qwen (28 layers).

## The Multi-Factor Design to Isolate Sycophancy

To disentangle signals, we use a 2×2 factorial design:

| Condition | User Ideology | Answer Agrees? | What’s Captured |
| :--- | :--- | :--- | :--- |
| **A** | Liberal | **Yes** (agrees) | User encoding + Topic + **+Sycophancy** |
| **B** | Liberal | **No** (disagrees) | User encoding + Topic + **-Sycophancy** |
| **C** | Conservative | **Yes** (agrees) | User encoding + Topic + **+Sycophancy** |
| **D** | Conservative | **No** (disagrees) | User encoding + Topic + **-Sycophancy** |

### Mathematical Isolation
$$ \text{Sycophancy Signal} = \frac{(A + C) - (B + D)}{2} $$

Why this works:
*   **User Identity**: (A+B) vs (C+D) cancels out.
*   **Topic/Question Bias**: Present in all, cancels out.
*   **Answer Token**: Balanced 50/50 in all groups, cancels out.
*   **Sycophancy**: (+1) - (-1) **Accumulates**.

## FAQ: Why Doesn't This Just Measure Token Identity?
**User Concern**: "If we extract at the answer token, aren't we just measuring the difference between embedding(A) and embedding(B)?"

**Answer**: Yes, for any *single* pair, the difference is dominated by the token identity (A vs B). However, that is why **A/B Balancing** is strictly required.

Imagine the activation vector $V$ is a sum of independent components:
$$ V = V_{token} + V_{behavior} $$

*   **Scenario 1 (Sycophancy is A)**: $V_1 = V_{(A)} + V_{syco}$
*   **Scenario 2 (Sycophancy is B)**: $V_2 = V_{(B)} + V_{syco}$

When we average over the dataset (where Scenario 1 and Scenario 2 occur equally often):
$$ Mean(V_{syco}) = \frac{(V_{(A)} + V_{syco}) + (V_{(B)} + V_{syco})}{2} = \frac{V_{(A)} + V_{(B)}}{2} + V_{syco} $$

We compare this against the **Non-Sycophantic** average:
$$ Mean(V_{non-syco}) = \frac{(V_{(B)} + V_{non}) + (V_{(A)} + V_{non})}{2} = \frac{V_{(A)} + V_{(B)}}{2} + V_{non} $$

**The Difference**:
$$ \Delta = Mean(V_{syco}) - Mean(V_{non-syco}) $$
$$ \Delta = \left( \frac{V_{(A)} + V_{(B)}}{2} + V_{syco} \right) - \left( \frac{V_{(A)} + V_{(B)}}{2} + V_{non} \right) $$
$$ \Delta = V_{syco} - V_{non} $$

The **Token Identity term** ($\frac{V_{(A)} + V_{(B)}}{2}$) is identical in both averages and **cancels out completely**, leaving only the behavioral difference.

## Critique 2: The Off-Distribution / "Surprise" Problem
**Critique**: "Forcing the model to give the 'Non-Sycophantic' answer might put it in an off-distribution state. If the model *really* wants to be sycophantic, forcing it to disagree might just capture a 'Surprise' or 'Error' vector rather than a 'Truthfulness' vector."

**Response**: This is a valid theoretical concern. However, in practice:
1.  **Linearity Assumption**: We assume the representation space is populated by directions closer to or further from the concept. Even if the model assigns low probability to the non-sycophantic answer, the *representation* of that answer still exists in the residual stream.
2.  **Symmetry**: We force *both* directions. We force it to Agree (Condition A) and Disagree (Condition B). Ideally, both are equally "forced" in terms of token generation, just with different semantic contents.
3.  **Validation is Key**: If the extracted vector was just "Surprise", it wouldn't generalize. The fact that these vectors can steer the model on *new, held-out questions* (as shown in CAA and Rimsky et al.) proves they capture the underlying behavioral feature, not just a transient OOD artifact.

## Critique 3: Is it just "General Agreeableness"?
**Critique**: "Are we measuring political sycophancy, or just the tendency to say 'Yes' / agree with whatever is presented?"

**Response**: The factorial design helps, but doesn't fully eliminate this if the "Agree" condition is always semantically aligned with the prompt.
*   **Mitigation**: This is why **Cross-Domain Evaluation** is standard. We extract the vector on Political questions, but we test its effect on *Philosophy* or *NLP* questions (as per the datasets). If the vector causes sycophancy in a totally different domain, it suggests we've found a general "Sycophancy" mechanism rather than just "Political Agreement".

## Critical Validation: PCA / Geometric Diagnostics

### 1. PCA Visualization
Project activations to 2D using PCA. Two types of clustering can emerge:
*   **Letter clustering**: Points separate by token (A vs B). **BAD (Confound)**.
*   **Behavioral clustering**: Points separate by target behavior (Syco vs Non-Syco). **GOOD (Signal)**.

Both clusterings exist. The goal is to verify that they lie on *different* axes.

### 2. Geometric Diagnostics
*   **Dimensionality (TwoNN)**: Is sycophancy 1D or multi-D?
*   **Clustering (HDBSCAN)**: Are there subtypes? Does it match ideology?
*   **Boundary Geometry**: Is the decision boundary linear?

## Summary of Recommendations
1.  **A/B Balance**: Strict 50/50 per label.
2.  **Pair Prompts**: Using the factorial design.
3.  **Extract at Answer**: Position of the first completion token.
4.  **Average Many Pairs**: 300+ pairs to cancel noise.
5.  **Mid-Layer Focus**: Layers 10-24.
6.  **PCA Validation**: Verify behavioral axis ≠ letter axis.

## References
*   **CAA**: Rimsky et al., 2023 ("Steering Llama 2 via Contrastive Activation Addition")
*   **Sycophancy Dataset**: Perez et al., 2022
*   **HyperSteer**: Sun et al., 2025
