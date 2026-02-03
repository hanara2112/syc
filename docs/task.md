# ICML Submission Roadmap: Sycophancy Dissociation

## Phase 1: Foundations & Geometry (Week 1)
- [x] Formal geometric decomposition ($v_{syc}, v_{part}$ calculation)
- [x] Layerwise signal curve (norm vs layer)
- [x] 2D geometric projection plot
- [x] Logit projection analysis (L14-L27)
- [x] Gradient sensitivity sweep (L14-L27)
- [ ] Local Gradient Sensitivity calculation ($\frac{\partial \text{logit}}{\partial \alpha}$)

## Phase 2: Causal Dissociation Proof (Week 2)
- [ ] Implement `run_steering_sweep.py`
- [ ] Compare steering effect (Flip Rate) at Layer 20 vs Layer 30
- [ ] Implement `activation_patching.py`
- [ ] Compare 1D steering vs 7680D patching effect at Layer 20
- [ ] Calculate AUROC/Decodability across all layers (Linear Probes)

## Phase 3: Generalization & Replication (Week 3)
- [ ] Cross-topic generalization (Political $v_{syc}$ on Math/Logic sycophancy)
- [ ] Replicate Core results (Geometry + Causal Window) on Llama-3-8B
- [ ] Replicate Core results on Mistral-7B

## Phase 4: Synthesis & Writing (Week 4)
- [ ] Draft ICML Abstract
- [ ] Prepare 6 Required Figures
- [ ] Draft Mechanisms section (Logit-insensitive vs Nonlinear)
- [ ] Final Paper Review
