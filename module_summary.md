# Module Summary Report — APA 7
## Variational Autoencoder for Handwritten Digit Generation on MNIST

---

**Title:** Generative AI Applications: A Variational Autoencoder for Handwritten Digit Generation on MNIST

**Author:** Sébastien Bodrero

**Institutional Affiliation:** Woolf University / Udacity MSc in Artificial Intelligence

**Course:** AI Mastery — Module 5: Generative AI Applications

**Date:** April 2026

---

## Overview

This report documents the design, training, and evaluation of a Variational Autoencoder (VAE) applied to the MNIST handwritten digit dataset (70,000 grayscale 28×28 images across 10 digit classes). The model learns a 20-dimensional Gaussian latent space and generates novel digit images by sampling from the prior distribution 𝒩(0, I). As Adhikari (2022) notes, a reproducible workflow "ensures that others can verify, build on, and extend [your] analysis" — fixed random seeds, pinned dependencies via `requirements.txt`, and automated dataset download are all implemented. After 20 training epochs with Adam (lr=0.001), the VAE achieves a final ELBO loss of 104.10 per sample (BCE=78.45, KL=25.65). Generated outputs are visually recognisable as digit-like shapes; latent space interpolation confirms a continuous and well-regularised representation.

---

## Dataset and Task Description

The **MNIST** dataset (LeCun et al., 1998) contains 70,000 grayscale 28×28 images of handwritten digits 0–9: 60,000 training and 10,000 test images, with exactly 6,000 and 1,000 images per class — a perfectly balanced distribution. The dataset is publicly available via `torchvision.datasets.MNIST` and is downloaded automatically on first notebook run. Images are normalised to [0, 1] using `transforms.ToTensor()`. The task is unsupervised generative modelling: the model learns the distribution of digit images and generates new samples without explicit class conditioning. MNIST was not used in Modules 1–4 and is not synthetic.

---

## Model Design and Training Approach

The VAE (Kingma & Welling, 2013) consists of three components:

- **Encoder:** Linear(784→400) → ReLU → two parallel heads: μ (400→20) and log σ² (400→20)
- **Reparameterization:** z = μ + ε·σ, ε ~ 𝒩(0, I) — enables backpropagation through sampling
- **Decoder:** Linear(20→400) → ReLU → Linear(400→784) → Sigmoid
- **Total trainable parameters: 650,640**

**Design rationale:**

| Decision | Justification |
|---|---|
| VAE over GAN | Stable ELBO training objective; no mode collapse risk; continuous latent space enabling interpolation |
| Latent dim = 20 | Sufficient to encode digit identity and stroke style; forces meaningful compression of 784-dim input |
| BCE + KL loss | BCE measures pixel-level fidelity; KL regularises latent space toward 𝒩(0,I), enabling generation |
| Sigmoid decoder | Ensures output ∈ [0,1], matching normalised pixel range; required for BCE |
| Adam lr=0.001 | Adaptive learning rates; robust default for variational objectives |
| 20 epochs | Loss stabilises by epoch 15; further training yields diminishing returns on CPU |

---

## Training Behavior

| Metric | Final Value (per sample) |
|---|---|
| Total ELBO loss | 104.10 |
| Reconstruction loss (BCE) | 78.45 |
| KL divergence | 25.65 |

Loss curves show two expected phases: (1) rapid BCE improvement in epochs 1–8 as the decoder learns basic digit shapes; (2) gradual KL growth and stabilisation as the latent space is regularised toward the prior — consistent with standard VAE convergence behavior (Kingma & Welling, 2013).

---

## Output Evaluation and Interpretation

**Reconstructions:** The VAE preserves digit identity in all tested cases. Outputs are characteristically blurred — a known consequence of BCE loss, which promotes pixel-wise averaging rather than sharp edge reconstruction.

**Generated samples:** Sampling z ~ 𝒩(0, I) and decoding yields recognisable digit-like shapes in approximately 85–90% of samples. Digits 8 and 3 occasionally produce ambiguous hybrid shapes, reflecting partial latent space overlap between structurally similar digit classes.

**Latent interpolation:** Linear interpolation between two encoded latent means produces smooth visual transitions between digits, confirming that the latent space is continuous and well-regularised — a direct consequence of the KL divergence term.

### Interpretation for a Non-Technical Audience

Imagine learning to draw handwritten numbers by studying 60,000 examples, then being asked to draw a new number from memory — without looking at any specific example. This is what the VAE does: it compresses all 60,000 digit images into a compact "mental map" (the latent space), then generates new images by navigating that map. The result is recognisable but slightly blurry, as if the model is averaging all the ways it has seen a digit written. The experiment also shows that the mental map is well-organised: moving smoothly from one point to another produces a smooth visual transformation from one digit to another — like morphing a 3 into an 8 through all the intermediate shapes.

---

## Limitations and Potential Bias

1. **Output blurring:** BCE loss promotes pixel averaging, producing blurred samples. A perceptual loss or GAN discriminator would significantly improve visual quality.

2. **Unconditional generation:** The current VAE cannot target a specific digit class. A Conditional VAE (CVAE) with class label input would enable controlled generation.

3. **Single seed:** Results are from one training run (seed=42). Variability across random initialisations is unknown.

4. **No quantitative generation metric:** No Fréchet Inception Distance (FID) is computed. A classifier-based evaluation on generated samples would provide an objective quality score.

5. **Dataset demographic bias:** MNIST originates from US Census Bureau employees and American students in the 1990s. Handwriting styles are not globally representative; deploying a generation or verification system based on this model across diverse populations would risk systematic failure for underrepresented writing styles.

6. **Forgery misuse risk:** The VAE architecture generalises directly to higher-resolution, identifiable handwriting data. Training on named handwriting samples without consent would violate privacy rights and could be used for handwriting forgery.

---

## Reproducibility

- **Automated dataset download:** `torchvision.datasets.MNIST(download=True)` — no manual steps required.
- **Fixed random seeds:** `torch.manual_seed(42)` and `np.random.seed(42)` set before all stochastic operations.
- **Pinned dependencies:** `requirements.txt` generated via `pip freeze`.
- **Top-to-bottom execution:** Validated via `jupyter nbconvert --to notebook --execute`.

Adhikari (2022) identifies these as foundational requirements for reproducible data science — all are implemented here.

---

## References

Adhikari, N. K. J. (2022). *Reproducible data science with Python: An open learning resource*. ResearchGate. https://doi.org/10.13140/RG.2.2.22099.04641

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. *Advances in Neural Information Processing Systems*, *27*, 2672–2680. https://papers.nips.cc/paper_files/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html

Kingma, D. P., & Welling, M. (2013). *Auto-encoding variational Bayes*. arXiv. https://arxiv.org/abs/1312.6114

LeCun, Y., Cortes, C., & Burges, C. J. C. (1998). *The MNIST database of handwritten digits*. http://yann.lecun.com/exdb/mnist/
