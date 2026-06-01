---
id: 3792699
published: true
tags:
- embeddings
- computervision
- selfsupervisedlearning
- paperreview
---



# [Paper Notes] JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture[🔗](https://arxiv.org/abs/2301.08243)

> **TL;DR:** I-JEPA learns a a generalized semantic representation with less data pairs by **predicting missing information in the embedding space**, which helps it disregard unnecessary noisy from input(pixel)-level details and learns at a **higher abstraction level** with good semantic generalization.

---

## 1. Innovation & Significance

* **The Bottleneck:** 
  * Image-text data pair labels are hard to find
  * Pixel level pre-training paired & data augmentation are strongly biased towards trained data distribution, hard to determine proper generalization and level of abstraction.
  * JEA's (Joint Embedding Architecture) collapse problem: encoder & decoder attempts to cheat by always landing on trivial constant when predicting itself (reconstruction) and gets away with an easy Error=0.
* **The Solution:**

> **Chain-of-thought**<br>
> ⭕ Mask pre-training to reduce data & generalize<br>
> ↓<br>
> ❌ Bad/lower semantic representation without semantic target, could be learning noisy local pixel correlation<br>
> ↓<br>
> ⭕ Learn at the embedding level to omit pixel input and generalize<br>
> ⭕ Adds context encoder & positional encoding to inject context  and force model to pick up image inherent structure from reconstructing multiple masked patches with one target.<br>
> ↓<br>
> ❌ JEAs wants to cheat: if I always map all pixels to a constant for both the predictor and end target encoder then the reconstruction error is always collapsed to zero! Hehe~<br>
> <img src="https://raw.githubusercontent.com/PsyDak-Meng/DakDevDiary/main/content/posts/Paper%20Notes/Computer%20Vision/JEPA/images/JEAs.png" alt="alt text"><br>
> ↓<br>
> ⭕ EMA (Exponential moving avg.): Update target encoder parameters from the EMA of context encoders. This 'delays' the target encoder to prevent collapsing (a trick from the BYOL paper[2020], proven essential to training JEAs with ViT).

## 2. Model & High-Level Intuitions
### 2.1 Model Architecture

![alt text](https://raw.githubusercontent.com/PsyDak-Meng/DakDevDiary/main/content/posts/Paper%20Notes/Computer%20Vision/JEPA/images/JEPA.png)

**Input:** randomly samples block masks from original image within certain aspect ratio changes, and apply mask for context image
### 2.1.2 Context
**Context Encoder:** ViT encodes context image to embedding {% katex inline %}S_x{% endkatex %}<br>**Mask Token**: an [1,D] randomly initialized shared learnable vector {% katex inline %}M{% endkatex %}, values are used where it is a masked pactc (colored pacthes in the figure).<br>**Positional embedding**: [1,D] sinusoidal embedding {% katex inline %}PE{% endkatex %}<br>**Predictor**: standard ViT, inputs masked token, context embedding and positional embedding for
{% katex %}
g_{\phi}(M + PE + S_x) = \hat{y}
{% endkatex %}

### 2.1.3 Target
**Target Encoder**: input original image and masked bbox to get corresponding embedding patches as {% katex inline %}y{% endkatex %}.

### 2.1.4 Loss & Training
The avg. {% katex inline %}L_2{% endkatex %} distance between predicted {% katex inline %}\hat{y}{% endkatex %} and target {% katex inline %}y{% endkatex %}.
{% katex %}
\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{(i,j) \in \mathcal{M}} \left| g_{\phi}(f_{\theta}(x_v),\ PE_{i,j}) - f_{\xi}(x)_{i,j} \right|^2
{% endkatex %}

Where:
{% katex inline %}f_{\theta}{% endkatex %} — context encoder (trained), processes visible patches {% katex inline %}x_v{% endkatex %}

{% katex inline %}f_{\xi}{% endkatex %} — target encoder (EMA), processes full image {% katex inline %}x{% endkatex %}

{% katex inline %}g_{\phi}{% endkatex %} — predictor (trained), takes context representations + positional embedding of masked position

{% katex inline %}\mathcal{M}{% endkatex %} — set of masked patch positions

{% katex inline %}PE_{i,j}{% endkatex %} — positional embedding at masked position {% katex inline %}(i,j){% endkatex %}
<hr>
Target encoder's EMA (parameters update not in loss by gradient descent, but direct update after each step):
{% katex %}
\xi_t = \alpha \xi_{t-1} + (1-\alpha)\theta_{t-1}
{% endkatex %}
{% katex %}
= (1-\alpha)\sum_{k=0}^{t} \alpha^k \theta_{t-k}
{% endkatex %}

The weight of a past context encoder snapshot {% katex inline %}\theta_{t-k}{% endkatex %} decays as {% katex inline %}\alpha^k{% endkatex %} — exponentially in how many steps ago it was. That's where the name comes from.

So with {% katex inline %}\alpha=0.996{% endkatex %}:

1 step ago: weight {% katex inline %}= 0.996^1 = 0.996{% endkatex %}<br>100 steps ago: weight {% katex inline %}= 0.996^{100} \approx 0.67{% endkatex %}<br>1000 steps ago: weight {% katex inline %}= 0.996^{1000} \approx 0.02{% endkatex %}



### 2.2 Intuitions
For anyone familiar with the encoder/decoder architecture pre-training, this paper's biggest innovation no doubt goes into it's self-supervision, the way of obtaining semantic information without labels. It jumped outside of 2 boxes:
* mask reconstruction is 1:1
* learning semantics needs labels.

### 2.2.1 Discover Image Semantics in the Context vs. Multi-Target Structure
It recognized the semantics exist in pixel structure and provide it to the model by multi context-to-target relationship (similar to data augmentation) combined with positional embedding. Essentially saying, given the context, here is what it misses in different locations, now the pixel & location variation of different mask patches in relation to the context image becomes the source of semantic information.
### 2.2.2 Target Encoder as Semantics Filter for Mask Reconstruction
While the context-to-target relation provides semantic, it learns low level semantic as it is equivalent to maximizing mutual information between the original and reconstructed image. But pixel data has high entropy from irrelevant details and reconstruction in pixel space never optimize towards compressing information.<br>Hence, I-JEPA's target encoder acts as a stochastic bottleneck, discarding unpredictable information from the target. In information theory,
{% katex %}
\max\ I(\hat{z};\ f_{\xi}(x))
{% endkatex %}
Rather than maximizing {% katex inline %}I(\hat{z};\ x){% endkatex %} directly. Since {% katex inline %}f_{\xi}(x){% endkatex %} already has low-level entropy compressed away, the predictor only needs to capture what's semantically predictable — the mutual information that survives the encoder bottleneck.