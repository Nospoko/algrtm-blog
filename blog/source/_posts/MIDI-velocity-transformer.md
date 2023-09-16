---
title: MIDI Velocity Prediction with Transformer
date: 2023-09-11
tags:
author: wmatejuk
---

### Why is it important?

{% algrtmImgBanner MIDI-velocity-transformer/banner-1.jpg pianoroll %}

MIDI velocity is a crucial element in music dynamics, determining the force with which a note is played, 
which profoundly influences the emotional quality of music. 

If you were to take a sequence of notes and predict their velocities by an untrained model, this is what you wold have ended up with:
<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2;">
    
  **Original** 
        
  **Predicted by untrained model**

</div>


<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2;">

  {% algrtmImg MIDI-velocity-transformer/samples/9115-real.png pianoroll 170px %}
  {% algrtmAudio MIDI-velocity-transformer/samples/9115-real.mp3 %}
        
  {% algrtmImg MIDI-velocity-transformer/samples/9115-pred-untrained.png pianoroll 170px %}
  {% algrtmAudio MIDI-velocity-transformer/samples/9115-pred-untrained.mp3 %}

</div>

Sounds pretty bad, doesn't it?

Our Transformer-based model aims to decode 
this nuanced aspect of musical expression, unraveling the hidden patterns 
within quantized MIDI data.



### Model Overview
{% algrtmImgBanner MIDI-velocity-transformer/transformer.png transformer%}
    
*The Transformer model* is ideal for this task because it excels at capturing complex dependencies in sequential data, making it well-suited for predicting MIDI velocities accurately.

This model's suitability arises from its *self-attention* mechanism,
which enables it to weigh the importance of different parts of the input sequence,
regardless of their temporal order.
In the context of MIDI data, this means that the Transformer can effectively learn
and leverage complex relationships between musical notes, their timing,
and how these factors influence the resulting velocity.

Strong prediction results signify the model's proficiency in extracting vital
features and comprehending intricate relationships. 
Its accurate encoding of quantized MIDI data and precise velocity predictions
mark a significant stride toward the realm of emotionally resonant AI music generation.
### Data Preprocessing
#### MIDI data
MIDI data describes notes by 5 features:
   1. Pitch - Represented as a number between 0 and 127 (or 21 to 108 for piano keys, reflecting the standard 88-key keyboard).
   2. Start - Indicates the moment a key is pressed, measured in seconds.
   3. End - Marks the second when the key is released.
   4. Duration - calculated as the time elapsed between the key's press and release.
   5. Velocity - ranging from 0 to 128, indicating the intensity of the key press.

#### Quantization
The way more suitable for quantization was what we engineered from this data.
We used 4 values to describe notes:
1. Pitch - same as above.
2. Dstart - time elapsed after start of previous note.
3. Duration - same as above,
4. Velocity - same as above.

This way we can achieve a more general data quantization, without having to consider
variations in the song's tempo or using bins that cover the entire duration of 
the sample. This approach ensures consistent quantization regardless of
the length of the composition.

We extracted 128-note samples which we quantized using 3 bins for dstart, 3 for duration and 3 for velocity.
Pitch information remained the same.

Here are **bin edges** we used:
```
dstart:
  - 0.0
  - 0.048177
  - 0.5
duration:
  - 0.0
  - 0.145833
  - 0.450000
velocity:
  - 0.0
  - 57.0
  - 74.0
  - 128.0
```
### Quantization Samples
<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2;">
    
  **Original** 
        
  **Quantized**

</div>

Franz Schubert / Franz Liszt: *Song Transcriptions: Aufenthalt, Gretchen am Spinnrade, Standchen von Shakespeare, Der Erlkonig*

<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2;">

  {% algrtmImg MIDI-velocity-transformer/samples/5002-real.png pianoroll 170px %}
  {% algrtmAudio MIDI-velocity-transformer/samples/5002-real.mp3 %}
        
  {% algrtmImg MIDI-velocity-transformer/samples/5002-quantized.png pianoroll 170px %}
  {% algrtmAudio MIDI-velocity-transformer/samples/5002-quantized.mp3 %}

</div>



Ludwig van Beethoven:  *Sonata No. 8 in C Minor, Op.13*
<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2;">

  {% algrtmImg MIDI-velocity-transformer/samples/9127-real.png pianoroll 170px %}
  {% algrtmAudio MIDI-velocity-transformer/samples/9127-real.mp3 %}
        
  {% algrtmImg MIDI-velocity-transformer/samples/9127-quantized.png pianoroll 170px %}
  {% algrtmAudio MIDI-velocity-transformer/samples/9127-quantized.mp3 %}

</div>

### Model Architecture

A transformer built as described in [Attention is all you need](https://arxiv.org/abs/1706.03762) paper was used.
The important hyperparameters:
- Number of layers in encoder and decoder: **6**
- Nuber of heads in attention layers: **8**
- Dimension of encoder and decoder outputs: **512**
- Dimension of a hidden layer of position-wise fast-forward network from each layer of encoder and decoder: **2048**


### Training and Evaluation
#### Data
   The model was trained on ~200 hours of musical data from 
   [roszcz/maestro-v1](https://huggingface.co/datasets/roszcz/maestro-v1) dataset containing 1276 pieces of 
   classical music performed during piano competition. 
#### Hardware and schedule
   Training on (very old) Nvidia GeForce GTX 960M with 4096 MiB of memory for 5 epochs (2723 steps) took only 7,5 hours.
   Each step took ~6 seconds.
#### Optimizer
Optimizer and learning rate were used as described in
[Attention is all you need](https://arxiv.org/abs/1706.03762) paper:
- Adam optimizer with *β1 = 0.9, β2 = 0.98* and *ϵ = 10−9*.
- The learning rate varied over the course of training, according to the formula:
*lrate = d_model^(-0.5) \* min(step_num^(−0.5), step_num \* warmup_steps^(−1.5))*

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_steps = 3000. 
#### Results
{% algrtmImgBanner MIDI-velocity-transformer/banner-2.jpg pianoroll %}
The model reaches **2.57 loss** and **5.13 average distance** between prediction and real value.
In contrast - untrained model has a **4.9 loss** and **30.7 average distance**
### Demonstration
#### Samples
<div style="-webkit-column-count: 3; -moz-column-count: 3; column-count: 3;">
    
  **Original** 

  **Predicted**

</div>

Johann Sebastian Bach, *Prelude and Fugue in D-sharp Minor*

<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2;">

  {% algrtmImg MIDI-velocity-transformer/samples/7177-real.png pianoroll 170px %}
  {% algrtmAudio MIDI-velocity-transformer/samples/7177-real.mp3 %}
        
  {% algrtmImg MIDI-velocity-transformer/samples/7177-predicted.png pianoroll 170px %}
  {% algrtmAudio MIDI-velocity-transformer/samples/7177-predicted.mp3 %}

</div>

Frédéric Chopin, *Etude Op. 10 No. 12*

<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2;">

  {% algrtmImg MIDI-velocity-transformer/samples/5171-real.png pianoroll 170px %}
  {% algrtmAudio MIDI-velocity-transformer/samples/5171-real.mp3 %}
        
  {% algrtmImg MIDI-velocity-transformer/samples/5171-predicted.png pianoroll 170px %}
  {% algrtmAudio MIDI-velocity-transformer/samples/5171-predicted.mp3 %}

</div>

#### Pieces with predicted velocity
Here are whole pieces from our dataset wirh original and predicted velocities

<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2;">
    
  **Original** 
        

  **Predicted**

</div>

Johann Sebastian Bach, *Prelude and Fugue in A-flat Major, WTC I*

<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2;">

  {% algrtmImg "MIDI-velocity-transformer/pieces/Johann Sebastian Bach Prelude and Fugue in A-flat Major, WTC I-pred.png" pianoroll 170px %}
  {% algrtmAudio "MIDI-velocity-transformer/pieces/Johann Sebastian Bach Prelude and Fugue in A-flat Major, WTC I.mp3" %}
        
  {% algrtmImg "MIDI-velocity-transformer/pieces/Johann Sebastian Bach Prelude and Fugue in A-flat Major, WTC I-pred.png" pianoroll 170px %}
  {% algrtmAudio "MIDI-velocity-transformer/pieces/Johann Sebastian Bach Prelude and Fugue in A-flat Major, WTC I-pred.mp3" %}

</div>

Wolfgang Amadeus Mozart, *Sonata in B-flat Major, K*

<div style="-webkit-column-count: 2; -moz-column-count: 2; column-count: 2;">

  {% algrtmImg "MIDI-velocity-transformer/pieces/Wolfgang Amadeus Mozart Sonata in B-flat Major, K-pred.png" pianoroll 170px %}
  {% algrtmAudio "MIDI-velocity-transformer/pieces/Wolfgang Amadeus Mozart Sonata in B-flat Major, K-pred.mp3" %}
        
  {% algrtmImg "MIDI-velocity-transformer/pieces/Wolfgang Amadeus Mozart Sonata in B-flat Major, K-pred.png" pianoroll 170px %}
  {% algrtmAudio "MIDI-velocity-transformer/pieces/Wolfgang Amadeus Mozart Sonata in B-flat Major, K-pred.mp3" %}

</div>

### Conclusion

### Acknowledgments and References

### Contact and Feedback