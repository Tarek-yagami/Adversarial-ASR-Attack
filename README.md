# Adversarial Attacks on Speech and Vision Models

White-box adversarial attacks against three pretrained models across two modalities: **Wav2Vec2**
(speech-to-text), **ResNet-18** (image classification), and **YOLOv8n** (object detection). Every
attack is Projected Gradient Descent (PGD) under an L∞ perturbation budget, verified end to end on
real audio/image samples with the transcriptions, predictions, and perturbation metrics committed
alongside the code — not just the method, but the measured result of running it.

## 📁 Repository structure
```
audio/    Wav2Vec2 ASR attack (PGD + psychoacoustic masking + energy-based filtering)
image/    ResNet-18 classification attack and YOLOv8n detection attack (PGD)
```
Each folder has its own notebook(s), a `samples/` directory with committed before/after
audio or images plus a `results.json`, and a README with the full results table and methodology.

## 📊 Highlights
- **Wav2Vec2**: PGD on the CTC loss, with the perturbation re-projected onto the epsilon ball after
  psychoacoustic masking and energy filtering are applied — an earlier version of this attack let
  post-processing silently blow past the stated epsilon by 5-6x. See [`audio/`](audio/).
- **ResNet-18**: an 8/255 (barely visible) perturbation flips top-1 predictions to 100% confidence in
  the wrong class — `espresso` becomes `Irish setter`. See [`image/`](image/).
- **YOLOv8n**: a disappearance attack reliably removes the true detection in every run, and also
  surfaces a real failure mode of untargeted attacks on detectors — the model hallucinates confident,
  nonexistent objects instead. See [`image/`](image/).

## 🔧 Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Tarek-yagami/Adversarial-ASR-Attack.git
   cd Adversarial-ASR-Attack
   ```
2. Install dependencies (one `requirements.txt` covers both `audio/` and `image/`):
   ```bash
   pip install -r requirements.txt
   ```
   `pydub` (used by the audio notebook) also requires [ffmpeg](https://ffmpeg.org/download.html) on
   your `PATH`.
3. Run either notebook — see [`audio/README.md`](audio/README.md) or [`image/README.md`](image/README.md)
   for usage and full results.

## 📖 References
- **PGD / Adversarial Training** - [Madry et al., 2017](https://arxiv.org/abs/1706.06083)
- **Wav2Vec2** - [Baevski et al., 2020](https://arxiv.org/abs/2006.11477)
- **Adversarial Attacks on ASR** - [Carlini & Wagner, 2018](https://arxiv.org/abs/1801.01944)
- **Adversarial Examples for Object Detection** - [Xie et al., 2017](https://arxiv.org/abs/1703.08603)

## License
[Apache License 2.0](LICENSE)
