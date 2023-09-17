---
title: 'Exploring VQ-VAEs'
date: 2023-09-17
tags: ['MIDI', 'ECG', 'Autoencoder', 'VQ-VAE']
author: Samuel
---


## Introduction

{% algrtmImgBanner VQ-VAEs/piano-hand.png piano %}

VQ-VAEs are a type of generative model that merge the strengths of traditional autoencoders and vector quantization, providing a more robust and efficient method for data compression and generation. [More on VQ-VAEs](#vq-vaes-a-brief-overview)

In this blog post, we'll delve into my recent research efforts centered on using VQ-VAEs for encoding 1D data—specifically, Electrocardiogram (ECG) readings and Musical Instrument Digital Interface (MIDI) data.

## The Need for Data Encoding
{% algrtmImgBanner VQ-VAEs/random-code.png code %}
#### Importance of 1D Data
1D data, especially in the realms of ECG and MIDI, carry crucial information. For instance, ECG data holds vital clues to a person's heart health, while MIDI sequences encode the soul of a musical piece. These data types are typically dense, capturing complex time-series information in a single dimension. As such, they are difficult to compress and store efficiently.

#### Existing Solutions and Their Limitations
Several methods are available for encoding 1D data, each with its own merits and drawbacks:

- **Autoencoders:** These neural network-based techniques are generally effective for many types of data but can sometimes sacrifice details for the sake of compression.

- **Fourier Transforms:** A mathematical staple for time-series data, yet not always suited for capturing non-linear or transient aspects of the data.

- **Smoothing/Averaging:** Simple but potentially lossy, as they can blur important details in the data.

While these methods have their uses, they also have limitations that one needs to consider depending on the application at hand. **Vector Quantized Variational Autoencoders** (VQ-VAEs) offer another avenue for exploration in the context of 1D data encoding, adding to the existing repertoire of techniques.


## VQ-VAEs: A Brief Overview
{% algrtmImgBanner VQ-VAEs/himeji-overview.png overview %}

#### Differences from Traditional Autoencoders

Traditional autoencoders consist of two primary components: an encoder that compresses input data into a lower-dimensional latent space, and a decoder that aims to reconstruct the original data from this compressed representation. Both parts are trained simultaneously with the objective of minimizing reconstruction error.

However, an intriguing issue arises from this architecture: sometimes the encoder doesn't capture as much information in the latent space as one might expect. This is because decoders, especially those with high capacity, can become exceedingly proficient at "filling in the gaps," or reconstructing missing or ambiguous information. As a result, the encoder might not learn a rich or informative latent space. Instead, the decoder compensates for the encoder's shortcomings, essentially becoming too good at its job for the encoder to improve.

#### How VQ-VAEs Address This Issue
VQ-VAEs introduce an additional layer of complexity with vector quantization, which addresses this issue. In VQ-VAEs, the encoder's output is not used directly for decoding. Instead, it is mapped to the nearest vector in a predefined codebook. This discrete representation, sourced from the codebook, is then used by the decoder for reconstruction.

What this achieves is a more balanced relationship between the encoder and the decoder:

- **Richer Latent Space:** Because the encoder's output is quantized to specific codebook vectors, it is forced to create a more informative latent representation to minimize the quantization error.

- **More Controlled Reconstructions:** The use of a codebook makes the latent space discrete, allowing for more consistent and controlled reconstructions, as each code vector has a specific meaning or representation.

- **Reduced Decoder "Cleverness":** The codebook acts as a form of "regularization," preventing the decoder from becoming too adept at reconstructing from inadequate encodings, thereby encouraging the encoder to be more descriptive.

## Diving into the Results
{% algrtmImgBanner VQ-VAEs/diving.png overview %}

Having explored the nuances of VQ-VAEs, it's time to delve into their practical applications. We applied the VQ-VAE model to two types of 1D data: ECG and MIDI, aiming to explore how well these models perform in real-world scenarios. Let's jump straight into the outcomes.

### Results: ECG Data
Our experimentation with ECG data yielded promising results in terms of both compression and reconstruction quality. The model was able to capture critical features like R-peaks and QRS complexes, essential for cardiac diagnosis.

{% algrtmImg VQ-VAEs/ecg-vqvae-reconstruction.png ecg %}
### Results: MIDI Data
MIDI data poses a unique challenge: it's a complex representation where even tiny nuances can significantly influence how we perceive the music. As such, it's difficult to quantify the quality of reconstructions. However, we can still evaluate the model's performance by comparing the original and reconstructed data.

It's worth noting that the pitch information was not encoded in our model; it's a part of the visualization to provide a more comprehensive understanding of the reconstructed data.

**Fragment of Robert Schumann Sonata No. 2 in G Minor, Op. 22:**


*Original:*

{% algrtmImg VQ-VAEs/original_pianoroll_schumann.png original 220px %}
{% algrtmAudio VQ-VAEs/schumann-original.mp3 %}


*Reconstruction:*

{% algrtmImg VQ-VAEs/reconstructed_pianoroll_schumann.png reconstructed 220px %}
{% algrtmAudio VQ-VAEs/schumann-reconstructed.mp3 %}

The visualizations and audio were generated using [fortepyan](https://github.com/Nospoko/fortepyan).