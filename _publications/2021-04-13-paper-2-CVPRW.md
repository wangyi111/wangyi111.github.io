---
title: "Machine-learned 3D Building Vectorization from Satellite Imagery"
collection: publications
excerpt: 'A novel method to recontruct 3D buildings from satellite images in Level of Detail (LoD) 2.'
venue: 'CVPR Workshop'
authors: 'Yi Wang, Stefano Zorzi, Ksenia Bittner'
paperurl: 'https://arxiv.org/abs/2104.06485'
permalink: /publications/2021/paper-2-CVPRW/
# citation: 'Your Name, You. (2010). &quot;Paper Title Number 2.&quot; <i>Journal 1</i>. 1(2).'
---

We propose a machine learning based approach for automatic 3D building reconstruction and vectorization. Taking a single-channel photogrammetric digital surface model (DSM) and panchromatic (PAN) image as input, we first filter out non-building objects and refine the building shapes of input DSM with a conditional generative adversarial network (cGAN). The refined DSM and the input PAN image are then used through a semantic segmentation network to detect edges and corners of building roofs. Later, a set of vectorization algorithms are proposed to build roof polygons. Finally, the height information from the refined DSM is added to the polygons to obtain a fully vectorized level of detail (LoD)-2 building model. We verify the effectiveness of our method on large-scale satellite images, where we obtain state-of-the-art performance.


[Wang, Y., Zorzi, S., & Bittner, K. (2021). Machine-learned 3D Building Vectorization from Satellite Imagery. arXiv preprint arXiv:2104.06485.]()