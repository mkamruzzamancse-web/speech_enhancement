# Reservoir-computing speech enhancement

Speech enhancement with an **Echo State Network (ESN)**, evaluated on held-out speakers and noise
against classical baselines (spectral subtraction, Wiener) and an optional GRU.
Rebuilt from my Master's research at Tottori University (*Speech Enhancement based on RNN using
Reservoir Computing*), with a corrected methodology and evaluation.

> **Status:** code complete and unit-tested; results table below is **to be filled in** from
> real runs on real speech. Numbers from the synthetic smoke test must not be reported.

## Method

Each utterance is converted to STFT log-magnitude frames `u_t` (16 kHz, 32 ms window, 8 ms hop).
The reservoir runs **over the frames of one utterance** (state reset per utterance):

```
x_t = (1 - a) x_{t-1} + a · tanh(W_in u_t + W x_{t-1})
y_t = W_out [x_t ; u_t ; 1]
W_out = (FᵀF + λI)⁻¹ FᵀY            (closed-form ridge regression, F = [X U 1])
```

* `W` is sparse and rescaled to spectral radius ρ (< 1 for the echo state property heuristic)
* `y_t` is the clean log-magnitude frame; the waveform is rebuilt with the **noisy phase**
  (magnitude capped at the noisy magnitude)
* optional look-ahead via `--context k` (stacks frames t-k … t+k)

## Evaluation protocol

* **Speaker-disjoint** train/val/test (one folder per speaker)
* **Noise-disjoint**: each noise file is split in time 70/15/15 into train/val/test portions
* Mixing at controlled **SNR** (default 0, 5, 10 dB), fixed seeds
* Metrics on **time-domain waveforms** at 16 kHz: SI-SDR, STOI, wideband PESQ
* Baselines: spectral subtraction, Wiener filter, GRU (same features and look-ahead)
* Ablations on the **validation** split: spectral radius, leak rate, reservoir size, ridge λ

## Install

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest -q tests
```

## Data

Not included. Suggested public sources (check their licences and download them yourself):

* clean speech, one folder per speaker: e.g. LibriSpeech or VoiceBank-DEMAND
* noise: NOISEX-92 (`white.wav`, `pink.wav`, `babble.wav`); the **file name is the noise label**

```
clean_dir/<speaker_id>/*.wav
noise_dir/white.wav  pink.wav  babble.wav
```

Smoke test without any data (synthetic, not speech):

```bash
python scripts/make_synthetic_data.py --out data_synth
python scripts/run_experiment.py --clean_dir data_synth/clean --noise_dir data_synth/noise \
       --n_res 300 --n_test_files 3 --snrs 0 10 --out runs/smoke
```

## Run

```bash
python scripts/run_experiment.py --clean_dir CLEAN --noise_dir NOISE --out runs/exp1 --gru
python scripts/ablate_esn.py     --clean_dir CLEAN --noise_dir NOISE --out runs/ablation
```

Outputs: `config.json`, `results.csv` (per utterance), `results.md` (mean ± std table), `esn.pkl`.

## Results

| noise | SNR (dB) | method | SI-SDR (dB) | STOI | PESQ-WB |
|---|---|---|---|---|---|
| _to be filled from `runs/exp1/results.md`_ | | | | | |

## Known limitations

* Noisy-phase reconstruction caps achievable quality
* Baseline noise PSD estimator assumes roughly stationary noise
* A reservoir may not beat a trained GRU; report whichever result you actually get, including
  accuracy-versus-training-cost, which is the usual argument for reservoir computing

## Original thesis code

The earlier notebooks (ESN / LSM, white / pink / babble) are kept in `legacy/` for reference.
They train and test on the same data and compute PESQ on spectrogram vectors, so their numbers
should not be compared with this repository's results.
