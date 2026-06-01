---
title: 'Paper Reading Notes: [JEPA]'
tags:
  - embeddings
  - computervision
  - selfsupervisedlearning
  - paperreview
published: true
id: 3792040
date: '2026-06-01T02:26:23Z'
---



# [Paper Notes] JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture[🔗](https://arxiv.org/abs/2301.08243)

> **TL;DR:** JEPA learns a a generalized semantic representation with less data pairs by **predicting missing information in the embedding space**, which helps it disregard unnecessary noisy from input(pixel)-level details and learns at a **higher abstraction level** with good semantic generalization.

---

## 1. Innovation & Significance

* **The Bottleneck:** 
  * Image-text data pair labels are hard to find
  * Pixel level pre-training paired & data augmentation are strongly biased towards trained data distribution, hard to determine proper generalization and level of abstraction.
  * JEA's (Joint Embedding Architecture) collapse probelm: encoder & decoder attempts to cheat by always landing on trivial constant when predicting itself (reconstruction) and gets away with an easy Error=0.
* **The Solution:**
    > **Chain-of-thought**  
    ⭕ Mask pre-training to reduce data & generalize  
    ↓  
    ❌ Bad/lower semantic representation without semantic target, could be learning noisy local pixel correlation  
    ↓  
    ⭕ Learn at the embedding level to omit pixel input and generalize  
    ⭕ Adds context encoder & positional encoding to inject context  and force model to pick up image inherent structure from reconstructing multiple masked patches with one target.  
    ↓  
    ❌ JEAs wants to cheat: if I always map all pixels to a constant for both the predictor and end target encoder then the reconstruction error is always collapsed to zero! Hehe~  
    ![alt text](https://raw.githubusercontent.com/PsyDak-Meng/DakDevDiary/main/content/posts/Paper%20Notes/Computer%20Vision/JEPA/images/JEAs.png)  
    ↓  
    ⭕ EMA (Exponential moving avg.): Update target encoder parameters from the EMA of context encoders. This 'delays' the target encoder to prevent collapsing (a trick from the BYOL paper[2020], proven essential to training JEAs with ViT). 

## 2. Model & High-Level Intuitions
### 2.1 Model Architecture
![alt text](https://raw.githubusercontent.com/PsyDak-Meng/DakDevDiary/main/content/posts/Paper%20Notes/Computer%20Vision/JEPA/images/JEPA.png)  
**Input:** randomly samples block masks from original image within certain aspect ratio changes, and apply mask for context image
### 2.1.2 Context
**Context Encoder:** ViT encodes context image to embedding ![S_x](https://latex.codecogs.com/png.image?S_x)  
**Mask Token**: an [1,D] randomly initialized shared learnable vector ![M](https://latex.codecogs.com/png.image?M), values are used where it is a masked pactc (colored pacthes in the figure).  
**Positional embedding**: [1,D] sinusoidal embedding ![PE](https://latex.codecogs.com/png.image?PE)  
**Predictor**: standard ViT, inputs masked token, context embedding and positional embedding for  
![equation](https://latex.codecogs.com/png.image?g_%7B%5Cphi%7D%28M%20%2B%20PE%20%2B%20S_x%29%20%3D%20%5Chat%7By%7D)

### 2.1.3 Target
**Target Encoder**: input original image and masked bbox to get corresponding embedding patches as ![y](https://latex.codecogs.com/png.image?y). 

### 2.1.4 Loss & Training
The avg. ![L_2](https://latex.codecogs.com/png.image?L_2) distance between predicted ![\hat{y}](https://latex.codecogs.com/png.image?%5Chat%7By%7D) and target ![y](https://latex.codecogs.com/png.image?y).  
![equation](https://latex.codecogs.com/png.image?%5Cmathcal%7BL%7D%20%3D%20%5Cfrac%7B1%7D%7B%7C%5Cmathcal%7BM%7D%7C%7D%20%5Csum_%7B%28i%2Cj%29%20%5Cin%20%5Cmathcal%7BM%7D%7D%20%5Cleft%7C%20g_%7B%5Cphi%7D%28f_%7B%5Ctheta%7D%28x_v%29%2C%5C%20PE_%7Bi%2Cj%7D%29%20-%20f_%7B%5Cxi%7D%28x%29_%7Bi%2Cj%7D%20%5Cright%7C%5E2)

Where:  
![f_{\theta}](https://latex.codecogs.com/png.image?f_%7B%5Ctheta%7D) — context encoder (trained), processes visible patches ![x_v](https://latex.codecogs.com/png.image?x_v)

![f_{\xi}](https://latex.codecogs.com/png.image?f_%7B%5Cxi%7D) — target encoder (EMA), processes full image ![x](https://latex.codecogs.com/png.image?x)

![g_{\phi}](https://latex.codecogs.com/png.image?g_%7B%5Cphi%7D) — predictor (trained), takes context representations + positional embedding of masked position

![\mathcal{M}](https://latex.codecogs.com/png.image?%5Cmathcal%7BM%7D) — set of masked patch positions

![PE_{i,j}](https://latex.codecogs.com/png.image?PE_%7Bi%2Cj%7D) — positional embedding at masked position ![(i,j)](https://latex.codecogs.com/png.image?%28i%2Cj%29)
<hr>
Target encoder's EMA (parameters update not in loss by gradient descent, but direct update after each step):  
![equation](https://latex.codecogs.com/png.image?%5Cxi_t%20%3D%20%5Calpha%20%5Cxi_%7Bt-1%7D%20%2B%20%281-%5Calpha%29%5Ctheta_%7Bt-1%7D)  
![equation](https://latex.codecogs.com/png.image?%3D%20%281-%5Calpha%29%5Csum_%7Bk%3D0%7D%5E%7Bt%7D%20%5Calpha%5Ek%20%5Ctheta_%7Bt-k%7D)

The weight of a past context encoder snapshot ![\theta_{t-k}](https://latex.codecogs.com/png.image?%5Ctheta_%7Bt-k%7D) decays as ![\alpha^k](https://latex.codecogs.com/png.image?%5Calpha%5Ek) — exponentially in how many steps ago it was. That's where the name comes from.

So with ![\alpha=0.996](https://latex.codecogs.com/png.image?%5Calpha%3D0.996):

1 step ago: weight ![= 0.996^1 = 0.996](https://latex.codecogs.com/png.image?%3D%200.996%5E1%20%3D%200.996)  
100 steps ago: weight ![= 0.996^{100} \approx 0.67](https://latex.codecogs.com/png.image?%3D%200.996%5E%7B100%7D%20%5Capprox%200.67)  
1000 steps ago: weight ![= 0.996^{1000} \approx 0.02](https://latex.codecogs.com/png.image?%3D%200.996%5E%7B1000%7D%20%5Capprox%200.02)



### 2.2 Intuitions
For anyone familiar with the encoder/decoder architecture pre-training, this paper's biggest innovation no doubt goes into it's self-supervision, the way of obtaining semantic information without labels. It jumped outside of 2 boxes:
* mask reconstruction is 1:1
* learning semantics needs labels.

### 2.2.1 Discover Image Semantics in the Context vs. Multi-Target Structure
It recognized the semantics exist in pixel structure and provide it to the model by multi context-to-target relationship (similar to data augmentation) combined with positional embedding. Essentially saying, given the context, here is what it misses in different locations, now the pixel & location variation of different mask patches in relation to the context image becomes the source of semantic information.
### 2.2.2 Target Encoder as Semantics Filter for Mask Reconstruction
While the context-to-target relation provides semantic, it learns low level semantic as it is equivalent to maximizing mutual information between the original and reconstructed image. But pixel data has high entropy from irrelevant details and reconstruction in pixel space never optimize towards compressing information.  
Hence, I-JEPA's target encoder acts as a stochastic bottleneck, discarding unpredictable information from the target. In informatio theory,  
![equation](https://latex.codecogs.com/png.image?%5Cmax%5C%20I%28%5Chat%7Bz%7D%3B%5C%20f_%7B%5Cxi%7D%28x%29%29)  
Rather than maximizing ![I(\hat{z};\ x)](https://latex.codecogs.com/png.image?I%28%5Chat%7Bz%7D%3B%5C%20x%29) directly. Since ![f_{\xi}(x)](https://latex.codecogs.com/png.image?f_%7B%5Cxi%7D%28x%29) already has low-level entropy compressed away, the predictor only needs to capture what's semantically predictable — the mutual information that survives the encoder bottleneck.
