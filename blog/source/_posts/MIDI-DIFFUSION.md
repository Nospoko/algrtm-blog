---
title: 'MIDI Diffusion Models'
date: 2023-09-25
tags: ['DDPM', 'Diffusion Models', 'Neural Networks', 'MIDI']
author: Jan
---

Repository: [MIDI Diffusion](https://github.com/Nospoko/midi-diffusion)

## Introduction
Diffusion Models (DDPM) have become go-to architecture for image generation tasks, but their capabilities in other data types are still highly unexplored.

In this research blog post we'll delve deeper into our recent research about applying Diffusion Models to MIDI data, specifically using conditioned DDPM to generate high quality velocity samples conditioned on low resolution dynamics of the recording.

## Diffusion Models
The most basic Diffusion Model architecture consists of two modules: forward diffusion which is used for noising data to some prior distribution and reverse diffusion, typically U-net style architecture which is trained to predict the amount of noise added.

Noise is predicted iteratively for better sample quality, which causes trade-off between sample quality and inference time. To tell model at which timestep it currently is, we inject it directly into the U-net after sinusoidal encoding.

### Toy diffusion
```python

```

Diffusion Models are extensively used for Text-to-Image tasks, such as: DALL-E 2, Midjourney, Imagen and Stable Diffusion. In next section we'll introduce Diffusion Model operating on MIDI data.

## DDPM for Velocity
MIDI files consist of four attributes needed that model will need to reconstruct to be able to generate them. These are: pitch, velocity, note start and note end. Start and end are absolute values which can cause problems for the model, so we instead convert them to representation that is easier to work with, but preserves all information. This representation is dstart and duration. Dstart corresponds to distance between start of current note and start of previous note. Duration corresponds to length of current note.


## DDPM for Velocity, Dstart and Duration
