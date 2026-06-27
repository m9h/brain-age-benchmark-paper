# M/EEG Brain Age Benchmark — Extended Setup Notes

**Date:** 2026-03-28
**Base repo:** https://github.com/meeg-ml-benchmarks/meeg-brain-age-benchmark-paper
**Paper:** Engemann et al., NeuroImage 262 (2022), doi:10.1016/j.neuroimage.2022.119521

---

## 1. Original Benchmark Summary

### Datasets (4)

| Dataset | Modality | Subjects | Age Range | Sampling Rate | Source |
|---------|----------|----------|-----------|---------------|--------|
| Cam-CAN | MEG | 646 | 18.5–88.9 | 1000 Hz → 200 Hz | https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/ |
| LEMON | EEG | 227 | 20–77 | 1000 Hz → 200 Hz | http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html |
| CHBP | EEG | 282 | 18–68 | 200 Hz (native) | https://www.synapse.org/#!Synapse:syn22324937 |
| TUAB | EEG | 1,385 | 0–95 | Variable (≤250 Hz) | https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tuab |

### Models (6)

| Pipeline | Type | Approach |
|----------|------|----------|
| Filterbank-Riemann | Classical ML | Covariance matrices + Riemannian geometry → Ridge regression |
| Filterbank-Source | Classical ML | MNE source localization + log-diag features → Ridge |
| Handcrafted Features | Classical ML | 18 temporal/spectral features → Random Forest |
| ShallowFBCSPNet | Deep learning | Shallow CNN (braindecode) |
| Deep4Net | Deep learning | 4-layer deep CNN (braindecode) |
| Dummy | Baseline | Mean strategy |

### Preprocessing Pipeline

1. MNE-BIDS Pipeline: high-pass 0.1 Hz, low-pass 49 Hz, resample to 200 Hz, 10s epochs
2. AutoReject: automated artifact rejection (cv=5)
3. Re-referencing: average reference (EEG), none (MEG)
4. Source estimation (for filterbank-source): dSPM, aparc_sub atlas (83 labels)
5. Feature extraction: 7 frequency bands (0.1–49 Hz), covariances via FFT (n_fft=1024)

### Evaluation

- 10-fold stratified CV (shuffle=True, seed=42)
- Metrics: MAE (years), R²
- Deep learning: recording-level splits, window-averaged predictions

---

## 2. New Dataset: CMI Healthy Brain Network (HBN) EEG

### Overview

- **Source:** Child Mind Institute
- **Subjects:** 3,000+ participants, ages 5–21
- **EEG:** 128-channel EGI HydroCel Geodesic Sensor Net (+ Cz reference = 129 ch)
- **Sampling rate:** 500 Hz, bandpass 0.1–100 Hz
- **Eyetracking:** simultaneous iView-X Red-m (SMI) at 120 Hz
- **Format:** BIDS-compliant (.set and .bdf)
- **Total size:** ~2.1 TB across 11 releases
- **Phenotypic data:** age, sex, handedness, CBCL psychopathology dimensions (p-factor, internalizing, externalizing, attention)

### Tasks (6)

| Task | Type | Description |
|------|------|-------------|
| Resting State (RS) | Passive | Eyes open/closed |
| Surround Suppression (SuS) | Passive | ~3.6 min visual runs (2 runs) |
| Movie Watching (MW) | Passive | 4 short films |
| Contrast Change Detection (CCD) | Active | Visual perception/decision-making (3 runs) |
| Sequence Learning (SL) | Active | Button-press sequence learning |
| Symbol Search (SyS) | Active | WISC-IV processing speed subtest |

### Releases on OpenNeuro

| Release | OpenNeuro ID | Subjects | Size |
|---------|-------------|----------|------|
| R1 | ds005505 | 136 | 103 GB |
| R2 | ds005506 | 152 | 120 GB |
| R3 | ds005507 | 183 | 140 GB |
| R4 | ds005508 | 324 | 230 GB |
| R5 | ds005509 | 330 | 224 GB |
| R6 | ds005510 | 134 | 91 GB |
| R7 | ds005511 | 381 | 245 GB |
| R8 | ds005512 | 257 | 157 GB |
| R9 | ds005513 | 295 | 185 GB |
| R10 | ds005514/ds005515 | 295 | 160 GB |
| R11 | — | 295 | 220 GB |
| NC (non-commercial) | — | 458 | 251 GB |

### Download

Stored on TrueNAS at `/data/raw/hbn/eeg/` (NFS-mounted on DGX Spark).

```bash
# On TrueNAS:
mkdir -p /mnt/tank/shared/raw/hbn/eeg
aws s3 cp s3://fcp-indi/data/Projects/HBN/BIDS_EEG/ /mnt/tank/shared/raw/hbn/eeg/ --recursive --no-sign-request
```

### Loading with MNE-BIDS

```python
from mne_bids import BIDSPath, read_raw_bids

bids_path = BIDSPath(
    subject='NDARAB793GL3',
    task='RestingState',
    root='/data/raw/hbn/eeg/cmi_bids_R1'
)
raw = read_raw_bids(bids_path)
```

### Relevance to Brain Age Benchmark

- Extends age range to pediatric/adolescent (5–21), complementing adult datasets
- Adds task diversity beyond resting state
- 3,000+ subjects makes it the largest single dataset in the benchmark
- Same BIDS format, compatible with existing MNE-BIDS pipeline

### References

- HBN-EEG FAIR preprint: bioRxiv 2024.10.03.615261v2 (Shirazi et al.)
- HBN Data Portal: http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/
- Original HBN paper: Nature Scientific Data 2017, doi:10.1038/sdata.2017.181

---

## 3. EEG2025 Ecosystem — NeurIPS 2025 EEG Foundation Challenge

### Challenge Overview

- **Competition:** "From Cross-Task to Cross-Subject EEG Decoding" (NeurIPS 2025)
- **Paper:** arXiv:2506.19141
- **Sponsors:** Meta, INRIA, UCSD, Donders Institute, Child Mind Institute
- **Scale:** 1,183 teams, 8,000+ submissions
- **Dataset used:** HBN-EEG (downsampled to 100 Hz, 0.5–50 Hz)

**Challenge 1 — Cross-Task Transfer Learning:** Predict response time from CCD task EEG. Winner: Team KUL_EEG (score 0.88668)

**Challenge 2 — Externalizing Factor Prediction:** Predict psychopathology score from EEG across tasks. Winner: Team JLShen (score 0.97843)

### Winner Models on HuggingFace (eeg2025/)

| Model | Files | Architecture |
|-------|-------|-------------|
| eeg2025/KU_Leuven (1st, Ch1) | challenge1_weights.pth (1.21 GB), challenge2_weights.pth (265 KB) | EEGTransformerFull — single-layer transformer with embedded preprocessing (CAR, Butterworth HP, resample 64 Hz, tanh artifact suppression). Code: github.com/corentinpuffay/neurips_challenge_2025_submission_model_2 |
| eeg2025/Sigma-Nova (2nd, Ch1) | weights_challenge_1.pt (2.61 GB), weights_challenge_2.pth (10.2 MB) | Unknown (no public code) |
| eeg2025/MIND-CICO (3rd, both) | 49 PyTorch model files (~245 MB total) | Ensemble of ~43 models |
| eeg2025/MBZUAI (2nd, Ch2) | attention_unet, inception, unet variants + sklearn meta-learners | U-Net variants + Attention U-Net + Inception + HGB/Ridge stacking |

---

## 4. Foundation Models Available for Brain Age Prediction

### In Braindecode (pip install braindecode)

Braindecode now ships 40+ architectures. Foundation/SSL models with pretrained weights:

| Model | Paper/Venue | Pretraining | Weights |
|-------|-------------|-------------|---------|
| **REVE** | NeurIPS 2025 | MAE on 25k subjects, 92 datasets, 60k+ hours | `brain-bzh/reve-base` (69M params), `brain-bzh/reve-large` (400M params) |
| **SignalJEPA** | SSL JEPA | Masked prediction, 16s windows | `braindecode/SignalJEPA` on HF |
| **BENDR** | Kostas et al. 2021 | BERT-style masked prediction | Yes |
| **BIOT** | Foundation model | Multi-dataset | HF Hub weights |
| **LaBraM** | Jiang et al. 2024 | Large Brain Model | In braindecode |
| **LUNA** | Doner et al. | Universal EEG embedding | In braindecode |
| **ContraWR** | Yang et al. 2021 | Contrastive learning | In braindecode |

Modern supervised architectures (not in original benchmark):

- EEGNeX, EEGConformer, EEGITNet, EEGTCNet, ATCNet, TSception
- MEDFormer, MSVTNet, PBT (Patched Brain Transformer), CTNet
- FBCNet, FBMSNet, SCCNet, SyncNet, TIDNet, SPARCNet
- EEGMiner, SincShallowNet, EEGSym, EEGInceptionMI, EEGInceptionERP

### External Foundation Models (standalone repos)

| Model | Stars | Venue | Architecture | Scale | Key Feature |
|-------|-------|-------|-------------|-------|-------------|
| **CBraMod** | 286 | ICLR 2025 | Criss-Cross Transformer | Custom | Weights: `weighting666/CBraMod` on HF |
| **Zuna** | 273 | — | 380M-param masked diffusion AE | 2M channel-hours | `pip install zuna`. Already in ~/dev/zuna-hf-wrapper |
| **NeuroGPT** | 219 | — | GPT-style | Multi-dataset | Foundation model for EEG |
| **NeuroLM** | 135 | ICLR 2025 | Multi-task bridge | Multi-dataset | Bridges language and EEG |
| **BrainOmni** | 59 | NeurIPS 2025 | BrainTokenizer + Transformer | Unified EEG+MEG | Only model handling both EEG and MEG jointly. HF: `sigureling/BrainOmni` |
| **EEGMamba** | 84 | Neural Networks 2025 | Mamba (state-space) | Custom | SSM architecture for EEG |
| **LEAD** | 85 | — | Foundation model | Alzheimer's data | First FM for Alzheimer's detection |
| **CSBrain** | 35 | NeurIPS 2025 Spotlight | Cross-scale spatiotemporal | Multi-scale | Spotlight paper |
| **REVE** | 22 | NeurIPS 2025 | MAE Transformer | 25k subj, 92 datasets | 4D positional encoding adapts to any electrode setup |
| **EEG-DLite** | 9 | AAAI 2026 | Data distillation | Distilled | Efficient FM training |
| **CodeBrain** | 8 | ICLR 2026 | Decoupled tokenizer + multi-scale | Multi-dataset | Latest approach |

### EEG Foundation Model Benchmark

**EEG-FM-Benchmark** (github.com/Dingkun0817/EEG-FM-Benchmark, 90 stars)

Benchmarks 12 open-source EEG foundation models across 13 datasets, 9 BCI paradigms. Key finding: "Linear probing is frequently insufficient; specialist models trained from scratch remain competitive." Paper: arXiv:2601.17883.

---

## 5. Additional Tools and Libraries

### EEGDash (pip install eegdash)

- Data-sharing platform: 500+ BIDS-compliant datasets, 27,053 participants, 25 labs
- Returns PyTorch Datasets compatible with braindecode
- Covers EEG, MEG, fNIRS, EMG, iEEG
- Docs: https://eegdash.org

### Starter Kit (EEG2025 Challenge)

- Repo: github.com/eeg2025/startkit
- Baseline: EEGNeX from braindecode (129 ch, 200 samples at 100 Hz)
- Training: AdamW, MSE/L1 loss, 2-second windows

---

## 6. Infrastructure Notes (DGX Spark)

### Storage Layout

| Location | Contents | Size |
|----------|----------|------|
| `/home/mhough/dev/meeg-brain-age-benchmark-paper/` | Benchmark code | ~50 MB |
| `/data/raw/hbn/eeg/` | HBN EEG BIDS data (NFS from TrueNAS) | ~2.1 TB |
| `/data/derivatives/hbn/` | FreeSurfer, autoreject, features, etc. | TBD |
| `/data/mhough/dmipy-data/` | dmipy data (moved from local) | 56 GB |
| `/data/mhough/dmipy-benchmarks/` | dmipy benchmarks (moved from local) | 4.4 GB |

### Compute

- GPU: 1x NVIDIA GB10 (128 GB unified memory)
- CPUs: 20 cores, 120 GB RAM
- Container: NGC PyTorch 26.02 (`nvcr.io/nvidia/pytorch:26.02-py3`)
- Slurm: `sbatch --gres=gpu:gb10:1`

### Kernel Issue (2026-03-27)

System was upgraded to kernel `6.17.0-1014-nvidia` but `linux-modules-nvidia-580-open` was not available at upgrade time. Currently running `6.17.0-1008-nvidia`. The nvidia-580-open module is now available:

```bash
sudo apt install linux-modules-nvidia-580-open-6.17.0-1014-nvidia
sudo reboot  # will boot into 1014 with working GPU
```

NFS fstab updated to use `x-systemd.requires=network-online.target` to prevent mount failures on reboot.

### Legion (FreeSurfer host)

- Lenovo Legion, 16-core x86_64, 8GB NVIDIA GPU
- Podman container: FreeSurfer 8.2.0-1 on Rocky Linux 9
- `/data` NFS-mounted from TrueNAS
- Use for: recon-all on HBN structural MRIs, source localization prep

---

## 7. HBN Structural MRI and FreeSurfer Derivatives

### MRI Data Locations (S3)

MRI and EEG are in **separate BIDS roots**. Subjects linked by NDAR IDs.

| Path | Description | Subjects |
|------|-------------|----------|
| `s3://fcp-indi/data/Projects/HBN/MRI/Site-CBIC/` | Raw BIDS MRI (CBIC site) | ~1,652 |
| `s3://fcp-indi/data/Projects/HBN/MRI/Site-RU/` | Raw BIDS MRI (Rutgers) | ~1,228 |
| `s3://fcp-indi/data/Projects/HBN/MRI/Site-CUNY/` | Raw BIDS MRI (CUNY) | ~686 |
| `s3://fcp-indi/data/Projects/HBN/MRI/Site-SI/` | Raw BIDS MRI (Staten Island) | ~345 |
| `s3://fcp-indi/data/Projects/HBN/BIDS_curated/` | Unified BIDS with session labels | Multi-site |

### Pre-computed FreeSurfer Derivatives (no need to re-run recon-all)

**Recommended: Reproducible Brain Charts (PennLINC)**
- Repo: https://github.com/ReproBrainChart/HBN_FreeSurfer
- Companion BIDS raw: https://github.com/ReproBrainChart/HBN_BIDS
- Curated with consistent pipelines (FreeSurfer + sMRIPrep)
- DataLad-managed — selective download of specific subjects/files
- Paper: Neuron 2025, doi:10.1016/j.neuron.2025.XX

```bash
mkdir -p /data/derivatives/hbn
cd /data/derivatives/hbn
datalad clone https://github.com/ReproBrainChart/HBN_FreeSurfer freesurfer
cd freesurfer
datalad get sub-NDAR*/surf/lh.white sub-NDAR*/surf/rh.white  # selective
```

**Legacy: FreeSurfer 6.0 on S3**
- Path: `s3://fcp-indi/data/Projects/HBN/derivatives/Freesurfer_version6.0.0/`
- ~287 subjects, full recon-all outputs (mri/, surf/, label/, stats/)

**Site-level: C-PAC pipeline outputs**
- Path: `s3://fcp-indi/data/Projects/HBN/MRI/Site-CBIC/derivatives/sub-*/T1w_HCP/freesurfer/`
- ANTs + FreeSurfer via C-PAC processing

### Additional HBN Derivatives on S3

| Path | Description |
|------|-------------|
| `Outputs/mindboggle_swf/` | Mindboggle morphometry (uses FreeSurfer input) |
| `Outputs/mriqc/` | MRI quality metrics |
| `BIDS_curated/derivatives/qsiprep/` | Diffusion preprocessing |
| `BIDS_curated/derivatives/afq/` | Automated Fiber Quantification |
| `CPAC_preprocessed/` | Functional preprocessing (C-PAC) |

### Planned Storage Layout

```
/data/raw/hbn/                       # raw BIDS data
├── eeg/                             # BIDS EEG (~2.1 TB) — downloading now
│   ├── cmi_bids_R1/
│   ├── cmi_bids_R2/
│   └── ...
└── mri/                             # Raw T1w BIDS (if needed later)
    ├── Site-CBIC/
    └── ...

/data/derivatives/hbn/               # processed outputs
├── freesurfer/                      # Pre-computed recon-all (RBC, DataLad)
│   ├── sub-NDARXX.../
│   └── ...
├── autoreject/
├── features/
└── mne-bids-pipeline/
```

Currently downloading to `/data/datasets/hbn-eeg/`, will reorganize to `/data/raw/hbn/eeg/` after download completes.

### Notes for Source Localization

- Brain-age benchmark filterbank-source pipeline only needs `fsaverage` template (ships with MNE)
- Individual FreeSurfer reconstructions needed for subject-specific source models
- FreeSurfer 8.2.0 on Legion available for any new recon-all runs via podman

---

## 8. Long-Term Vision: NeuroTechX Community Resource

### Goal

Build out a comprehensive, reproducible neuroimaging and BCI resource stack that NeuroTechX can host via **TechSoup.ca cloud compute support for nonprofits**. The local infrastructure (DGX Spark, TrueNAS, Legion) serves as the development/staging environment; the target is cloud-hosted community access.

### Assets Being Assembled

**Datasets (on TrueNAS, /data/datasets/):**
- HBN EEG: 3,000+ pediatric subjects, 128-ch, 6 tasks (~2.1 TB) → `/data/raw/hbn/eeg/`
- HBN FreeSurfer derivatives (via RBC DataLad)
- HBN structural MRI (available on S3 as needed)
- Cam-CAN, LEMON, CHBP, TUAB (from original benchmark)
- EEGDash: gateway to 500+ BIDS datasets, 27k participants

**Pre-trained Models:**
- REVE base/large (69M/400M params, 25k subjects, 92 datasets)
- BrainOmni (unified EEG+MEG)
- SignalJEPA, BENDR, BIOT, LaBraM, LUNA (braindecode)
- CBraMod (ICLR 2025)
- Zuna (380M-param masked diffusion AE)
- EEG2025 competition winners (KU Leuven, Sigma-Nova, MIND-CICO, MBZUAI)

**Benchmarking Framework:**
- Extended meeg-brain-age-benchmark with 5+ datasets, 15+ models
- EEG-FM-Benchmark integration (12 foundation models, 13 datasets)
- Reproducible via containers (NGC PyTorch, FreeSurfer podman)

**Compute:**
- Development: DGX Spark (GB10 GPU, 128GB unified), Legion (8GB NVIDIA, FreeSurfer)
- Storage: TrueNAS (58.2T pool, 54.6T free)
- Target: TechSoup.ca cloud compute for NeuroTechX nonprofit hosting

### References

- NeuroTechX: https://neurotechx.com/
- TechSoup Canada: https://www.techsoup.ca/
- EEGDash: https://eegdash.org
- Reproducible Brain Charts: https://reprobrainchart.github.io/
- EEG Foundation Model Benchmark: arXiv:2601.17883

---

## 9. TODO — Benchmark Extension Plan

### Phase 1: Setup
- [ ] Create venv/container with dependencies
- [ ] Download HBN EEG to TrueNAS (`/data/raw/hbn/eeg/`)
- [ ] Write `config_hbn.py` for benchmark repo
- [ ] Validate HBN loading with MNE-BIDS

### Phase 2: Preprocessing
- [ ] Run MNE-BIDS pipeline on HBN resting state data
- [ ] Run AutoReject
- [ ] Extract filterbank covariances and handcrafted features
- [ ] Validate channel consistency across subjects

### Phase 3: New Models
- [ ] Add REVE (base + large) via braindecode
- [ ] Add BrainOmni (EEG+MEG joint model)
- [ ] Add EEGConformer, ATCNet, EEGNeX
- [ ] Add Zuna as preprocessing step (neural denoising)
- [ ] Evaluate KU Leuven winner architecture (EEGTransformerFull)

### Phase 4: Evaluation
- [ ] Run all models on all 5 datasets (original 4 + HBN)
- [ ] Compare foundation model fine-tuning vs training from scratch
- [ ] Analyze pediatric vs adult age prediction performance
- [ ] Generate updated benchmark figures
