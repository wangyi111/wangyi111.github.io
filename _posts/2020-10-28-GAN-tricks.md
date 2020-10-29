---
title: 'GAN tricks'
permalink: /posts/2020/10/gan-tricks/
categories: programming
tags:
  - GAN
toc: true
toc_label: "CONTENT"
---

## 01: GAN's problem

Generally, GAN is a minimax problem[1]:

$$
\operatorname{minmax}_{C}[D, G)=\mathbb{E}_{\mathbf{x} \sim P_{r}(\mathbf{x})}[\log D(\mathbf{x})]+\mathbb{E}_{\mathbf{z} \sim P_{\mathbf{z}}(\mathbf{z})}[\log (1-D(G(\mathbf{z})))]
$$

For Discriminator, it is optimizing:

$$
-\mathbb{E}_{x \sim P_{r}}[\log D(x)]-\mathbb{E}_{x \sim P_{g}}[\log (1-D(x))]
$$

And for Generator:

$$
\mathbb{E}_{x \sim P_{g}}[\log (1-D(x))]
$$

For fixed Generator, Ian Goodfellow has proved the optimized Discriminator is:

$$
D_{G}^{*}(\mathbf{x})=\frac{P_{r}(\mathbf{x})}{P_{r}(\mathbf{x})+P_{g}(\mathbf{x})}
$$

Which means in ideal case it reaches Nash equilibrium: $$P_r = P_g =0.5$$.

Now if we add a term unrelated to G, the Generator loss becomes:

$$
\mathbb{E}_{x \sim P_{r}}[\log D(x)]+\mathbb{E}_{x \sim P_{g}}[\log (1-D(x))]
$$

We insert optimized D into this loss and can then acquire:

$$
L_G = 2 J S\left(P_{r} \| P_{g}\right)-2 \log 2
$$

which means we are actually optimizing the JS divergence between the two distributions. However in reality, $$P_r$$ and $$P_g$$ has large probability to have little overlap (i.e. low-dimensional manifolds in high-dimensional space has probably 0 measure). Therefore the JS divergence is almost a constant, meaning no gradient will backprograte to Generator.

For the performance of the experiment, we will see $$Loss_D$$ becomes 0 and $$Loss_G$$ stops updating.

To solve this problem, Ian Goodfellow provides another loss for Generator:

$$
\mathbb{E}_{x \sim P_{g}}[-\log (D(x))]
$$

But this raises another issue, as before we insert the optimized D into the loss, and will get:

$$
K L\left(P_{g} \| P_{r}\right)-2 J S\left(P_{r} \| P_{g}\right)
$$

There are two problems here:

1. JS and KL divergence have adverse signs, meaning they are conflict with each other;
2. KL divergence is asymmetric: 

$$
\begin{aligned}
&\text { If } p_{\text {data}}(x) \rightarrow 0 \text { and } p_{g}(x) \rightarrow 1, \text { we have }\\
&K L\left(p_{g} \| p_{d a t a}\right) \rightarrow+\infty\\
&\text { If } p_{\text {data}}(x) \rightarrow 1 \text { and } p_{g}(x) \rightarrow 0, \text { we have }\\
&K L\left(p_{g} \| p_{d a t a}\right) \rightarrow 0
\end{aligned}
$$

In the experiments, we will see either unstable loss changes or generated samples lacking diversity (mode collapse).

Due to the problems above, GAN's training is usually very difficult to adjust. Fortunately, reserches have found several tricks to help make it more stable to train GANs.

## 02: GAN tricks

### 2.1: Model choice

If no knowledge about model choice, choose DCGAN[2] or ResNet[3] as base model.

### 2.2: Input

If input image, normalize to [-1,1]; if input random noise, sample from N(0,1).

### 2.3: Output

Use 1x1 or 3x3 convolution with 1 or 3 channels as output layer, a suggested activation function is tanh.

### 2.4: Decoder

Choose upsample + conv2d instead of conv2dTranspose to avoid checkboard artifact. Use pixelshuffle[4] for super-resolution task. Use gated-conv2d[5] for image repairing task.

### 2.5: Normalization

Batch normalization is good for encoder (feature extraction), but for decoder (generation task) suggest using other methods: instance normalization[6], layer normalization[7] for parameterized methods and pixel normalization[8] for non-parameterized methods. If you don't know which to use, there's a switchable normalization[9] which combines all methods.

### 2.6: Discriminator

If you want to generate high-resolution image, use multi-stage discriminator[8]: maxpool input image to different scales and input to 3 discrimators (same structure but different net parameters).

For diversity of generated image, use mini-batch discriminator[10] (learn the similarity of generated samples); in PGGAN[11], calculate the statistics of feature maps as additional feature channel.

### 2.7: Label smoothing

Smooth the label of real samples to randomly 0.9-1.1 and keep fake samples same as 0.0, this could improve the stability of GAN training[10].

### 2.8: Instance noise

Add noise to real and fake samples to manually expand the two distributions, thus making them have overlap in high-dimensional space and JS divergence come to play again.

### 2.9: TTUR

Update D for several times and then update G; or set different learning rates for D and G to manually adjust their balance[12].

### 2.10: Exponential Moving Average

Apply moving average to history parameters to stabilize the training[13].

### 2.11: GAN loss

There are two family of losses: f-dviergence (including KL, JS divergence, etc.) and Integral Probability Metric (IPM). For the second family, the network learns the distance (or the effor to move one distribution to the other) between real and fake samples. A famous WGAN[14] uses Wasserstein distance (earth-mover distance) which solves the gradient vanishing and mode collapse problem at the same time with a simple implementation:
* remove sigmoid from the last layer of Discriminator
* don't use log for G and D losses
* each time after updating network weights, clip them into a fixed constant range
* don't use momentum based optimizer (e.g. Adam), use RMSProp or SGD

Another advantage of WGAN is it can give us a clear metric to judge the network's improvement (the smaller the W-distance, the better the generated result). However, in reality the clipped weights would go closer to the up/low boundary of the clipping range, which is later improved with WGAN-GP[15] with gradient penalty.

Some other GAN losses like LeastSquare[16] and Hinge loss[17,18] are also useful, and other auxiliary losses like L1, total variation[19], style[20], perception[19] are helpful with combination of GAN loss.

| Type | formula |
| ---- | ------- |
| GAN  | $$L_{D}^{G A N}=E[\log (D(x))]+E[\log (1-D(G(z)))]$$ |
|      | $$L_{G}^{G A N}=E[\log (D(G(z)))]$$ |
| LSGAN | $$\boldsymbol{L}_{D}^{L S G A N}=E\left[(D(x)-1)^{2}\right]+E\left[D(G(z))^{2}\right]$$ |
|       | $$L_{G}^{L S G A N}=E\left[(D(G(z))-1)^{2}\right]$$ |
| WGAN | $$L_{D}^{W G A N}=E[D(x)]-E[D(G(z))]$$ |
|      | $$L_{G}^{W G A N}=E[D(G(z))]$$ |
| WGAN-GP | $$\left.L_{D}^{W G A N_{-} G P}=L_{D}^{W G A N}+\lambda E[(\mid \nabla D(\alpha x+(1-\alpha) G(z))) \mid-1)^{2}\right]$$ |
|         | $$\left.L_{G}^{W G A N_{-} G P}=L_{G}^{W G A N}$$ |
| Hinge loss | $$L_{D}=E[\operatorname{relu}(1-D(x))]+E[\operatorname{relu}(1+D(G(z)))]$$ |
|            | $$-E[D(G(z))]$$ |


### 2.12: Spectral normalization

For the IPM loss family like WGAN, there's a K-Lipschitz constraint, which is dealt by clipping weights in WGAN. Reasearches later develop another method called spectral normalization[21] for solving this constraint, which is usually better than WGAN-GP.

### 2.13: PGGAN and coarse-to-fine

Progressive training strategy[8], train first small size or coarse image, and step-by-step go to fine level.

## 03: References

[1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.

[2] Radford, Alec et al. “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.” CoRR abs/1511.06434 (2016)

[3] He, Kaiming et al. “Deep Residual Learning for Image Recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016)

[4] Shi, Wenzhe et al. “Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016)

[5] Yu, Jiahui et al. “Free-Form Image Inpainting with Gated Convolution.” CoRRabs/1806.03589 (2018)

[6] Ulyanov, Dmitry et al. “Instance Normalization: The Missing Ingredient for Fast Stylization.” CoRR abs/1607.08022 (2016)

[7] Ba, Jimmy et al. “Layer Normalization.” CoRR abs/1607.06450 (2016)

[8] Karras, Tero et al. “Progressive Growing of GANs for Improved Quality, Stability, and Variation.” CoRR abs/1710.10196 (2018)

[9] Luo, Ping et al. “Differentiable Learning-to-Normalize via Switchable Normalization.” CoRRabs/1806.10779 (2018)

[10] Salimans, Tim et al. “Improved Techniques for Training GANs.” NIPS (2016)

[11] Demir, Ugur, and Gozde Unal. "Patch-based image inpainting with generative adversarial networks." arXiv preprint arXiv:1803.07422 (2018).

[12] Heusel, Martin et al. “GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.” NIPS (2017)

[13] Yazici, Yasin et al. “The Unusual Effectiveness of Averaging in GAN Training.” CoRRabs/1806.04498 (2018)



[14] Arjovsky, Martín et al. “Wasserstein GAN.” CoRR abs/1701.07875 (2017)

[15] Gulrajani, Ishaan, et al. "Improved training of wasserstein gans." Advances in neural information processing systems. 2017.

[16] Mao, Xudong, et al. "Least squares generative adversarial networks." Proceedings of the IEEE International Conference on Computer Vision. 2017.

[17] Zhang, Han, et al. "Self-attention generative adversarial networks." arXiv preprint arXiv:1805.08318 (2018)

[18] Brock, Andrew, Jeff Donahue, and Karen Simonyan. "Large scale gan training for high fidelity natural image synthesis." arXiv preprint arXiv:1809.11096 (2018).

[19] Johnson, Justin et al. “Perceptual Losses for Real-Time Style Transfer and Super-Resolution.” ECCV (2016)

[20] Liu, Guilin et al. “Image Inpainting for Irregular Holes Using Partial Convolutions.” ECCV(2018).

[21] Yoshida, Yuichi and Takeru Miyato. “Spectral Norm Regularization for Improving the Generalizability of Deep Learning.” CoRR abs/1705.10941 (2017)

[22] Gui, Jie, et al. "A review on generative adversarial networks: Algorithms, theory, and applications." arXiv preprint arXiv:2001.06937 (2020).

[23] https://www.chainnews.com/articles/042578835630.htm







