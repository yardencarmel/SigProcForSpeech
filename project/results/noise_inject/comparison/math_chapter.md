# Mathematical Analysis of Noise-Injection Style Transfer Methods

## Overview

This chapter provides a rigorous mathematical treatment of four methods for
acoustic style transfer in flow-matching TTS systems, specifically applied to
F5-TTS (Chen et al., ACL 2025).

All methods share a common foundation: they manipulate the inference process
of a Conditional Flow Matching (CFM) model without retraining, exploiting the
geometric structure of the ODE trajectory to transfer stylistic properties
from a reference speaker B to speech conditioned on speaker A's voice.

---

## Background: Conditional Flow Matching

F5-TTS trains a vector field $v_\theta(x, c, t)$ that transports samples from
a base distribution $p_0 = \mathcal{N}(0, I)$ to the data distribution $p_1$
of mel spectrograms, conditioned on a reference mel $c$.

The forward ODE is:
$$\frac{dx}{dt} = v_\theta(x_t, c, t), \quad x_0 \sim p_0, \quad t \in [0, 1]$$

During inference with classifier-free guidance (CFG):
$$v_\text{guided}(x, c, t) = v_\theta(x, \varnothing, t) + \lambda \left[ v_\theta(x, c, t) - v_\theta(x, \varnothing, t) \right]$$

where $\lambda$ is the CFG strength and $\varnothing$ is the unconditional (dropped) conditioning.

The goal of style transfer: generate speech with speaker A's voice characteristics
(identity, timbre) but carrying stylistic/emotional properties from speaker B
(prosody, energy contour, speaking rate).

---

## Method A — SDEdit Noise Injection

**Starting point modification only; vector field unchanged.**

### Formulation

$$x_0 = (1 - \alpha) \cdot \varepsilon + \alpha \cdot \text{mel}_B, \quad \varepsilon \sim \mathcal{N}(0, I)$$

The ODE integrates from $t_\text{start} = \alpha$ to $t = 1$, conditioned on $\text{mel}_A$:

$$\frac{dx}{dt} = v_\text{guided}(x_t, \text{mel}_A, t), \quad t \in [\alpha, 1]$$

### Geometric Interpretation

In flow matching, the trajectory from $x_0$ to $x_1$ passes through a
"noise-to-data" manifold. By biasing $x_0$ toward $\text{mel}_B$, we start
the ODE in a neighbourhood of B's mel in noise space. The identity conditioning
($\text{mel}_A$) then acts as a **gravitational pull** that re-orients the
trajectory toward A's data manifold.

The SDEdit connection: if $\alpha$ is small, the ODE sees mostly noise at
$t=\alpha$ and B's influence is mild; if $\alpha$ is large, B's structure
dominates and the ODE must "undo" it with only $1-\alpha$ integration time
available. This creates a tradeoff between **style bleeding** (high $\alpha$)
and **identity preservation** (low $\alpha$).

### Sway Sampling Interaction

Sway sampling modifies the timestep schedule:
$$t' = t + s \left( \cos\!\left(\frac{\pi}{2} t\right) - 1 + t \right)$$

Negative $s$ concentrates ODE steps near $t=0$ (the noisy end). Combined with
SDEdit, negative sway allocates more function evaluations to the critical
early-integration region where B's structure is being processed.

### Hyperparameters

- $\alpha \in [0, 1]$: blending weight; 0 = pure Gaussian, 1 = pure mel_B
- $s$: sway coefficient (typically $s \in [-1, 0]$)

---

## Method B — Style Guidance (2-Pass ODE Extrapolation)

**Vector field modification at every ODE step; starting point unchanged.**

### Formulation

At each ODE timestep $t$, the transformer is evaluated twice:

$$v_A(x, t) = v_\text{guided}(x, \text{mel}_A, t) \quad \text{(identity direction)}$$
$$v_B(x, t) = v_\text{guided}(x, \text{mel}_B, t) \quad \text{(style direction)}$$

The combined vector field is:
$$v_\text{style}(x, t) = v_A + g \cdot (v_B - v_A) = (1-g) \cdot v_A + g \cdot v_B$$

where $g$ is the **guidance scale**.

### Geometric Interpretation

This is a **linear extrapolation in vector-field space**. The direction
$v_B - v_A$ points from the identity trajectory toward the style trajectory.
Scaling it by $g$ controls how far along that direction we move:

- $g = 0$: pure identity (no style) — reduces to vanilla F5-TTS
- $g = 1$: equal blend of both fields
- $g > 1$: **extrapolation** beyond the style field (analogous to negative
  guidance in classifier-free guidance)

Crucially, this method never touches $x_0$ — style is injected entirely through
the velocity field. Because the velocity field is integrated continuously, the
effect is distributed across the full temporal extent of the mel spectrogram,
unlike Method A where style is concentrated at the starting frame.

### Analogy to Classifier-Free Guidance

Standard CFG extrapolates from unconditional to conditional:
$$v_\text{CFG} = v_\varnothing + \lambda (v_c - v_\varnothing)$$

Method B extrapolates from identity-conditional to style-conditional:
$$v_\text{style} = v_A + g (v_B - v_A)$$

The formal structure is identical, with the identity conditioning $\text{mel}_A$
playing the role of the "unconditional" baseline and $\text{mel}_B$ as the target condition.

### Computational Cost

2× transformer evaluations per ODE step → 2× wall-clock time vs baseline.
NFE stays the same; only the per-step cost doubles.

---

## Method C — Scheduled Conditioning Blend (Step Function)

**Time-varying conditioning; single pass per ODE step.**

### Formulation

The conditioning is switched at a user-specified $t^* \in [0, 1]$:

$$c(t) = \begin{cases} \text{mel}_B & \text{if } t < t^* \\ \text{mel}_A & \text{if } t \geq t^* \end{cases}$$

The ODE becomes:
$$\frac{dx}{dt} = v_\text{guided}(x_t, c(t), t)$$

### Geometric Interpretation

Flow matching ODE dynamics differ across the time axis:

- **Early steps** ($t$ small, near noise): the model resolves low-frequency
  spectral envelope, speaking rate, and prosodic shape — properties dominated
  by the training distribution of the conditioning signal.
- **Late steps** ($t$ large, near data): the model refines speaker-specific
  texture, fine formant structure, and micro-prosody.

By assigning $\text{mel}_B$ to early steps, we allow B's coarse prosodic
skeleton to inform the structural scaffold of the trajectory. By handing off to
$\text{mel}_A$ for late steps, we ask the model to "paint over" this scaffold
with A's acoustic identity.

### Key Properties

- **Single-pass**: same compute as vanilla F5-TTS
- **Non-smooth vector field**: the switch at $t^*$ creates a discontinuity in
  the conditioning, which the Euler ODE integrator handles without issue but
  may interact with higher-order solvers
- **Degenerate cases**: $t^*=0$ → always use A (identity baseline);
  $t^*=1$ → always use B (pure style, voice cloning from B)

### Extensions

The step function can be replaced by a smooth schedule:

*Linear:* $c(t) = (1-t) \cdot \text{mel}_B + t \cdot \text{mel}_A$

*Cosine:* $c(t) = \text{mel}_B + \frac{1 + \cos(\pi t)}{2} (\text{mel}_A - \text{mel}_B)$

These smooth schedules interpolate through the conditioning space, potentially
reducing the sharp acoustic discontinuity at $t^*$.

---

## Method D — Noise Statistics Transfer

**Starting point modification (statistics-only); vector field unchanged.**

### Formulation

$$x_0 = \frac{\varepsilon}{\|\varepsilon\|_\sigma} \cdot \sigma_\text{target} + \alpha \cdot \mu_B$$

where:
$$\sigma_\text{target} = \alpha \cdot \text{std}(\text{mel}_B) + (1-\alpha) \cdot \text{std}(\varepsilon)$$
$$\mu_B = \text{mean}(\text{mel}_B)$$

This rescales the Gaussian noise to have the same variance as a blend of
mel_B's spectral amplitude and native Gaussian variance, then adds a fraction
of mel_B's global mean offset.

### Comparison with Method A

| Property | Method A (SDEdit) | Method D (Stats) |
|---|---|---|
| x_0 construction | $(1-\alpha)\varepsilon + \alpha \cdot \text{mel}_B$ | rescaled $\varepsilon$ with B's stats |
| Temporal structure from B | Yes — B's frames appear at specific positions | No — only global amplitude shape |
| Maximum style leakage | Strong at high $\alpha$ | Weak; only envelope |
| WER degradation | Significant at $\alpha > 0.3$ | Expected milder |
| Interpretability | Direct frame blending | Spectral normalisation |

### Information-Theoretic View

Method D transfers the **first moment** ($\mu_B$) and **second moment** ($\sigma_B^2$)
of mel_B's amplitude distribution into the starting noise, while Method A
transfers up to **infinite-order statistics** (the exact joint distribution of
frames in mel_B). Method D is thus a strict information-theoretic lower bound
on how much style information can be injected via x_0 manipulation while
preserving the Gaussianity structure of the noise distribution.

---

## Comparison Summary

| Axis | Method A | Method B | Method C | Method D |
|---|---|---|---|---|
| Where style acts | $x_0$ (frames) | $v(x,t)$ (velocity) | $c(t)$ (condition) | $x_0$ (stats) |
| Temporal specificity | High (frame-level) | Distributed (all steps) | Step-coarse | None |
| Compute overhead | ~1× | ~2× | ~1× | ~1× |
| Identity conditioning | Pure (never blended) | Blended per step | Partial | Pure |
| Sway interaction | Strong | Moderate | Moderate | Mild |
| Primary control | $\alpha$ | $g$ | $t^*$ | $\alpha$ |

The richest information pathway is Method B (style enters at every ODE step
via the vector field), making it the most expressive but also the most
computationally expensive. Method A occupies a middle ground: B's temporal
structure seeds $x_0$ but identity conditioning dominates integration. Method C
creates a "coarse-to-fine" semantic switch without extra compute. Method D is
the lightest intervention, useful as an ablation to isolate the contribution
of amplitude statistics vs. temporal frame structure.

---

## References

1. Chen, S.-g., et al. (2025). F5-TTS: A Fairytaler that Fakes Fluent and
   Faithful Speech with Flow Matching. *ACL 2025*. arXiv:2410.06885.

2. Meng, C., et al. (2022). SDEdit: Guided Image Synthesis and Editing with
   Stochastic Differential Equations. *ICLR 2022*. arXiv:2108.01073.

3. Ho, J. & Salimans, T. (2022). Classifier-Free Diffusion Guidance.
   *NeurIPS 2022 Workshop on DGMs*. arXiv:2207.12598.

4. Lipman, Y., et al. (2023). Flow Matching for Generative Modeling.
   *ICLR 2023*. arXiv:2210.02747.

5. Chen, R. T. Q., et al. (2018). Neural Ordinary Differential Equations.
   *NeurIPS 2018*. arXiv:1806.07366.
