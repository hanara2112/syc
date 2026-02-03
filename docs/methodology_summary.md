# Activation Extraction Methodology

## Executive Summary
This document outlines the rigorous methodology for extracting sycophancy steering vectors from Large Language Models (LLMs) using the **Contrastive Activation Addition (CAA)** framework. The primary goal is to isolate the pure "sycophancy" signal—the tendency of the model to agree with the user's view—while mathematically canceling out confounding variables like prompt semantics, user persona encoding, and token identity.

## The Core Challenge: Confounds in Forced Completion
When we force a model to output a specific answer (e.g., "(A)"), the resulting activations contain a mixture of signals:
1.  **Token Identity**: "I am generating the token 'A'".
2.  **Answer Semantics**: "I am saying something conservative".
3.  **Prompt Encoding**: "The user is conservative".
4.  **Behavioral Signal**: "I am agreeing with the user" (Sycophancy).

Naive extraction (e.g., looking at "Liberal Prompts" vs "Conservative Prompts") fails because it conflates these factors. We need to isolate #4.

## The Solution: Factorial Design & A/B Balancing

We employ a **2x2 Factorial Design** combined with **A/B Token Balancing** to systematically cancel confounds.

### 1. Factorial Conditions
For every question, we construct four scenarios:

| Condition | User Persona | Model Answer | Agreement (Behavior) |
| :--- | :--- | :--- | :--- |
| **A** | Liberal | Liberal | **Agree (Sycophantic)** |
| **B** | Liberal | Conservative | **Disagree (Non-Sycophantic)** |
| **C** | Conservative | Conservative | **Agree (Sycophantic)** |
| **D** | Conservative | Liberal | **Disagree (Non-Sycophantic)** |

**Sycophancy Signal Formula**:
$$ \text{Vector}_{syco} = \frac{(A + C) - (B + D)}{2} $$

*   **User Identity Confound**: Cancels because Liberal ($A+B$) is balanced against Conservative ($C+D$).
*   **Answer Semantics Confound**: Cancels because Liberal Answer ($A+D$) is balanced against Conservative Answer ($B+C$).
*   **Remaining Signal**: The interaction term (Agreement), which accumulates.

### 2. A/B Token Balancing
To remove the "Token Identity" confound (the model representing "(A)" vs "(B)"), we ensure that within *each* condition, the specific answer token is randomized.
*   50% of "Liberal Answers" are the token "(A)".
*   50% of "Liberal Answers" are the token "(B)".
 This decorrelates the token identity from the semantic meaning and the behavioral agreement.

## Extraction Protocol

### 1. Extraction Position: The "First Answer Token"
**Critical**: We extract activations at the position of the **first token of the completion** (i.e., the " (" or "A" token).
*   *Why?* In a forced completion setup, the model doesn't "know" it's agreeing until it processes the forced token. Extracting at the last prompt token is ineffective because the inputs are identical for both "Agree" and "Disagree" completions up to that point. The difference only manifests *at* the answer token.

### 2. Layer Selection
Based on Rimsky et al. (CAA paper):
*   **Early Layers (0-10)**: Dominated by token identity (clustering by A/B).
*   **Middle-Late Layers (10-25)**: Behavioral signals (sycophancy) emerge and become linearly separable.
*   **Final Layers**: May be dominated by output formatting.
*   **Recommendation**: Extract from layers 10 through 24 (for a ~32 layer model like Qwen-7B).

### 3. Procedure
1.  **Input**: The `sycophancy_factorial_dataset.jsonl`.
2.  **Forward Pass**: Run the model on `prompt + completion`.
3.  **Extraction**: Save the hidden state vector at the index corresponding to the start of the completion.
4.  **Aggregation**: Group vectors by Condition (A, B, C, D).

## Geometric Validation
Before using the steering vector, verify its quality using PCA:
1.  **Project to 2D**: Run PCA on the extracted vectors.
2.  **Color by Behavior**: Points should cluster into "Sycophantic" vs "Non-Sycophantic".
3.  **Color by Token**: Points may also cluster by "(A)" vs "(B)".
4.  **Success Criterion**: The "Behavior" axis should be distinct (ideally orthogonal) to the "Token" axis. If they are aligned, the result is confounded.
