# Adversarial Attack on Wav2Vec2

## 📌 Description
This demonstrates an adversarial attack against the Wav2Vec2 ASR model.
By adding small, imperceptible perturbations to an audio signal, we can manipulate the model's transcription while keeping the modifications inaudible to humans.

## ⚡ Methods Implemented
- **Projected Gradient Descent (PGD)** attack on CTC loss
- **Psychoacoustic Masking** to hide perturbations
- **Energy-Based Filtering** to make the attack more discreet

## 🚀 Usage
From the repository root, with dependencies installed (see the [top-level README](../README.md)):
```bash
cd audio
jupyter notebook
```
Open `attack_wav2vec2.ipynb` and run the cells top to bottom. It downloads `facebook/wav2vec2-large-960h`
(~1.2GB, cached after the first run) and attacks the two sample clips under `samples/`.

## 📊 Results
Two short, license-free sample clips (synthetic text-to-speech audio, not a copyrighted recording or
anyone's voice) were run through the attack at two perturbation budgets each:

| sample | config | epsilon | WER | L∞ | L2 | SNR (dB) |
|---|---|---|---|---|---|---|
| sample1_pangram | mild | 0.015 | 0.000 | 0.0150 | 3.286 | 19.9 |
| sample1_pangram | strong | 0.04 | 0.200 | 0.0400 | 7.291 | 13.0 |
| sample2_tech | mild | 0.015 | 0.400 | 0.0150 | 3.831 | 18.9 |
| sample2_tech | strong | 0.04 | 0.400 | 0.0400 | 8.641 | 11.8 |

`WER` is the word error rate between the original and adversarial transcription (there's no external
ground-truth transcript, so this measures how much the attack degrades the model's own baseline reading
of each clip). `L∞`/`L2` are the perturbation size in the normalized `[-1, 1]` waveform, kept within the
stated `epsilon` budget by re-projecting after psychoacoustic masking and energy filtering are applied.
`SNR` is signal power over perturbation power, a rough proxy for how audible the perturbation is likely
to be.

Raw and adversarial `.wav` files for every sample/config pair are committed under `samples/` — open them
directly on GitHub to listen, or see `samples/results.json` for the full metrics.

## 📖 References
- **Wav2Vec2 Paper** - [https://arxiv.org/abs/2006.11477](https://arxiv.org/abs/2006.11477)
- **Adversarial Attacks on ASR** - [https://arxiv.org/abs/1801.01944](https://arxiv.org/abs/1801.01944)
