---
title: Datasets that Deliver
date: 2023-10-08
tags:
author: Samuel
---

## Comparison
We'll be comparing the following datasets:
A. **MyMaskedMidiDataset** from [midi-bert](https://github.com/Nospoko/midi-bert/blob/abc596e633128ca53ee0a1efaf53daa6e200a2ee/data/dataset.py#L83)
B. **MidiDataset** from [masked-midi-modeling](https://github.com/Nospoko/masked-midi-modeling/blob/7c22e7dd681822f8b7a0b5b7276b11b262bff6b8/data/dataset.py#L78)
both dataset need data in different format, here's an overview of how to create them

## The Data
To maintain consistency, our comparison will be based on the [maestro-v1-sustain](https://huggingface.co/datasets/roszcz/maestro-v1-sustain) dataset. Both datasets require data to be pre-processed in different formats..

#### Preparation time
**MyMaskedMidiDataset**:
Preparation took an average of `3min 39s` per loop, with the loop running 5 times.

**MidiDataset**:
In contrast, this dataset was speedier, clocking in at `18.9 s` per loop over 10 iterations.

The pre-processing times mentioned above were calculated on a test set, with the records having these metrics:
`Mean: 4188.757062146893, std: 3130.797449499849, max: 16966, min: 366, median: 3264.0`

## Benchmarking Dataset Performance
Head to [dataset benchmarking](https://github.com/Nospoko/dataset-benchmarking) for full code used for the benchmarking.



#### Performance insights
- **MyMaskedMidiDataset** performance:
{% algrtmImg datasets/midi-bert.png ecg 400px %}
<!-- ![midi-bert](../assets/datasets/midi-bert.png) -->
- **MidiDataset** performance:
{% algrtmImg datasets/masked.png ecg 400px %}
<!-- ![masked](../assets/datasets/masked.png) -->
As we can see, the trend suggests that with larger batch sizes, the performance of both datasets improves. However, the **MidiDataset** is consistently faster than **MyMaskedMidiDataset** by a factor of 4-5 times.


## TODO: tips?