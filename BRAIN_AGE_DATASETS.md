# Brain-Age M/EEG Datasets

Dataset catalog for the M/EEG brain-age benchmark. Two arms:
1. **Resting EEG/MEG** (Engemann-2022 replication core): LEMON, CHBP, TUAB, Cam-CAN, HBN —
   plus OpenNeuro/NEMAR additions (Part 3).
2. **Sleep EEG/PSG** (separate analysis arm, Part 1): test whether **sleep EEG is a stronger
   brain-age predictor than resting EEG**, pooling for large N.

---

## Part 1 — NSRR Sleep-EEG Datasets

A separate analysis arm from the Engemann-2022 resting-EEG replication: pool as many
National Sleep Research Resource (NSRR, sleepdata.org) polysomnography (PSG) cohorts as
possible to test whether **sleep EEG is a stronger brain-age predictor than resting EEG**,
and to get a much larger N for that estimate.

Compiled 2026-06-11. Primary source: NSRR FAIR paper (Zhang et al., *Sleep* 2024,
[PMC11236948](https://pmc.ncbi.nlm.nih.gov/articles/PMC11236948/), Table 1), cross-checked
against the 2018 JAMIA commons paper ([PMC6188513](https://pmc.ncbi.nlm.nih.gov/articles/PMC6188513/))
and individual `sleepdata.org/datasets/<slug>` montage pages. Numbers flagged ¹ are best
estimates to verify against live dataset pages before relying on them.

## Access reality (applies to ALL datasets below)

- There is **no open, no-approval tier**. Every PSG dataset requires its own
  **Data Access & Use Agreement (DAUA)**, reviewed and approved **per dataset**
  (typically a few days each), and the DAUA **expires after 3 years**.
- Having an sleepdata.org account is necessary but not sufficient — you must submit a
  separate "Request Data Access" on each dataset's page.
- After approval, a **single personal token** (https://sleepdata.org/token) works across
  all approved datasets. Store at `~/.nsrr_token` (never commit).
- Download tool: the NSRR ruby gem (`gem install nsrr --no-document`; Ruby 3.2.3 present
  on gx10-dgx-spark). Example: `nsrr download shhs/edfs/shhs1 --fast`.

## EEG-bearing datasets (raw EDF PSG with scalp EEG)

| Slug | Full name | N (subj / PSG) | Age / population | EEG montage | Fs | Access |
|------|-----------|----------------|------------------|-------------|----|--------|
| shhs | Sleep Heart Health Study | 5,804 / 8,444 (2 visits) | 40–89, adult/elderly community | C3/A2, C4/A1 | 125 Hz | DAUA |
| mros | Outcomes of Sleep Disorders in Older Men | 2,911 / 3,933 (V1+V2) | 65–89, elderly **men only** | C3/A2, C4/A1 | 256/512 Hz¹ | DAUA |
| mesa | Multi-Ethnic Study of Atherosclerosis (Sleep) | 2,237 / 2,056 | 54–95, multi-ethnic | **single C4/M1** | 256 Hz | DAUA |
| wsc | Wisconsin Sleep Cohort | 1,123 / 3,671 (longitudinal) | 37–85, adult community | C3/M2, C4/M1 | 200 Hz¹ | DAUA |
| cfs | Cleveland Family Study | 735 / 730 | 6–88, **family-based**, apnea-enriched | C3/M2, C4/M1 | 128/256 Hz¹ | DAUA |
| hchs | Hispanic Community Health Study / SOL | 16,415 / 12,088² | 18–76, Hispanic/Latino | mostly limited-montage HSAT² | — | DAUA (restricted) |
| stages | Stanford Technology Analytics & Genomics in Sleep | 1,881 / 2,055 | 13–84, sleep-clinic referrals | full AASM (C/F/O) | varies | DAUA |
| mnc | Mignot Nature Communications | 3,000 / 1,438 | 18–91, normals + narcolepsy/abnormal | C3,C4,F3,F4,O1,O2 | 128 Hz | DAUA |
| apples | Apnea Positive Pressure Long-term Efficacy Study | 1,516 / 1,104 | 18–84, OSA clinical (CPAP RCT) | full montage | — | DAUA (non-commercial) |
| nchsdb | NCH Sleep DataBank | 3,673 / 3,984 | 0–18(–58), **pediatric** clinical | high-density (Fp1/2,Fz,Cz,Pz,Oz,C3/4,F3/4,O1/2,T3/4) | 256 Hz | DAUA |
| chat | Childhood Adenotonsillectomy Trial | 1,243 / 1,639 | 5–9, **pediatric** OSA RCT | AASM pediatric | — | DAUA |
| ccshs | Cleveland Children's Sleep & Health Study | 517 / 515 | 16–19, **adolescent** | C3/A2, C4/A1 | 128 Hz¹ | DAUA |
| sof | Study of Osteoporotic Fractures | 461 / 453 | 65–89, elderly **women only** | C3/A2, C4/A1 | — | DAUA |
| haas | Honolulu-Asia Aging Study (sleep apnea) | 718 / 717 | 79–97, elderly Japanese-American men | central EEG | — | DAUA |
| heat | Heart Biomarker Evaluation in Apnea Treatment | 318 / 591 | 45–75, OSA clinical | EEG present | — | DAUA |
| numom2b | nuMoM2b (pregnancy outcomes) | 3,012 / 5,341² | 14–44, **pregnant women**, HSAT-heavy | limited montage² | — | DAUA |
| bestair | Best Apnea Interventions in Research | 169 / 518 | 46–76, OSA + CVD risk | EEG present | — | DAUA |
| abc | Apnea, Bariatric surgery & CPAP study | 49 / 132 | 26–64, obese OSA | EEG present | — | DAUA |
| msp | Maternal Sleep in Pregnancy & the Fetus | 106 / 106 | 18–42, **pregnant women** | EEG present | — | DAUA |
| sdbaem | SDB, ApoE & Lipid Metabolism | 712 / 712 | 13–90, mixed | EEG present | — | DAUA |
| fd | Forced Desynchrony (± chronic sleep restriction) | 28 / ~1,000 | 20–34, healthy lab/circadian | dense within-subject EEG | — | DAUA |
| nhp | NOP agonists in non-human primates | 5 / 10 | 14–19 (**primates**) | EEG | — | exclude |
| cf | Cox & Fell review sample | 5 / 3 | toy set | EEG | — | exclude |

¹ verify against live dataset montage page. ² HCHS/SOL and nuMoM2b are dominated by
limited-montage home sleep apnea tests (HSAT); usable-scalp-EEG count is much smaller than
the headline — treat as uncertain.

**Non-EEG NSRR datasets (excluded):** ecsiup, ansrs, shiecc, oya (actigraphy / questionnaire
/ HSAT only).

## Totals

- **Raw EEG-bearing subject sum:** ≈ 52,000 — but inflated by HCHS/SOL (16k) and nuMoM2b (3k)
  HSAT records without clean scalp EEG.
- **Realistic usable scalp-EEG pool:** ≈ **28,000–30,000 subjects** — matches the
  **26,673-individual / 13-cohort harmonized PSG release** NSRR itself ships (apples, ccshs,
  cfs, chat, mesa, mnc, mros, msp, nchsdb, shhs, sof, stages, wsc).
- **Adult-only usable scalp EEG:** ≈ 18,000–20,000. The four pillars alone (shhs 5.8k +
  mros 2.9k + mesa 2.2k + wsc 1.1k) ≈ **12,000**.

## Recommended pooling plan

**Tier 1 — clean adult core (~12k, request these first, in parallel):**
shhs, mesa, mros, wsc. Standard central EEG, wide adult age span. shhs is the top pick
(largest, two visits → longitudinal age delta). Caveats: mesa is single-channel C4/M1;
mros is men-only.

**Tier 2 — extend the young-adult tail (+~3k):**
cfs (6–88 in one montage, but family-clustered + apnea-enriched), stages (13–84 full
montage, clinic-referred, known duplicate EDFs), mnc (18–91, mixes normals + narcolepsy).

**Separate pediatric arm (maturation ≠ aging — model independently):**
nchsdb (0–18, high-density), chat (5–9), ccshs (16–19, almost no age variance).

**Exclude for adult brain-age:** sof/haas (single-sex, narrow elderly), apples/heat/bestair/abc
(heavy OSA pathology), numom2b/msp (pregnancy), fd (N=28, narrow 20–34), nhp/cf (non-human/toy),
hchs/numom2b (mostly HSAT, no clean scalp EEG).

## Benchmark design notes / open knobs

- **Montage heterogeneity:** shhs/mros/cfs are dual-channel C3+C4; mesa is single C4/M1;
  stages is full AASM. coffeine filterbank-Riemann needs a consistent channel set — either
  restrict to the common **C3/C4 (or just C4)** denominator, or fit per-cohort and harmonize
  in tangent space.
- **Sleep stage:** brain age likely differs by N2 vs N3 vs REM. Decide whether features are
  computed per-stage or whole-night. This is the scientific knob and the likely reason sleep
  EEG could beat resting EEG (far more data/subject + stage-specific aging signatures).
- **Sampling rate:** 125–256 Hz spread; resample to a common rate (Engemann fb pipeline
  already resamples).

## NSRR Sources

- NSRR FAIR paper (Sleep 2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11236948/ (Table 1)
- NSRR data commons (JAMIA 2018): https://pmc.ncbi.nlm.nih.gov/articles/PMC6188513/
- NSRR datasets index: https://sleepdata.org/datasets
- NSRR ruby gem: https://github.com/nsrr/nsrr-gem
- Token: https://sleepdata.org/token

---

## Part 2 — Already in the benchmark (for reference)

| Dataset | Modality | Condition | N | Source | Status |
|---------|----------|-----------|---|--------|--------|
| LEMON (MPI-Leipzig) | EEG | resting EO/EC | ~210 | MPI direct (also OpenNeuro **ds000221**) | staged, preprocessing running |
| CHBP (Cuban Human Brain Project) | EEG | resting | ~282 | Synapse `syn22324937` | token ready, not downloaded |
| TUAB (Temple Abnormal) | EEG | clinical resting | ~2.7k | Temple (credentialed) | pending credentials |
| Cam-CAN | MEG | resting | ~650 | MRC application | pending form |
| HBN (Healthy Brain Network) | EEG | pediatric resting | ~3k+ | eegdash / OpenNeuro ds005505-516 | partial |

---

## Part 3 — OpenNeuro / NEMAR additions

NEMAR (nemar.org) indexes the OpenNeuro EEG/MEG/iEEG corpus. **Every OpenNeuro dataset is
mirrored on a public (no-sign-request) S3 bucket** (`s3://openneuro.org/<ds>`), so these
need **no DUA, no token** — download immediately with `download_openneuro_dgx.sbatch`.
Compiled 2026-06-11 via the OpenNeuro GraphQL API + per-dataset `participants.tsv` on S3.

### Resting EEG additions (fills the "healthy adult-lifespan" gap that small LEMON leaves)

| Accession | Name | N | Age (in tsv) | Montage | Notes |
|-----------|------|---|--------------|---------|-------|
| **ds005385** | Dortmund Vital Study | **608** (208 re-tested) | **20–70**, mean 44 ✓ | 64-ch 10-20 @1kHz, EO+EC | **TOP add.** Healthy adult lifespan, standard montage, longitudinal re-test. ~3× LEMON's N |
| **ds003775** | SRM Resting-state (Oslo) | 111 | 17–71 ✓ | BioSemi 64-ch, eyes-closed | Clean healthy wide-adult, 2nd pick |
| ds005305 | Microstates & executive fn | 165 | 18–36 ✓ | — | young-adult skew |
| ds005280 | "223 By BP" resting | 223 | 16–32 ✓ | BrainProducts | student, narrow age |
| ds005292 | "142 by Biosemi" resting | 142 | ~18–22 ✓ | BioSemi | young, narrow |
| ds007358 | Field EEG India+Tanzania | ~2000 | 13–70+ **binned** ⚠ | 16-ch Emotiv consumer | huge N but coarse age + domain shift |
| ds005486 | PREDICT | 159 | unconfirmed ⚠ | — | verify age field |

### Sleep EEG additions (the HEALTHY anchor NSRR's clinical cohorts lack)

| Accession | Name | N | Age | Notes |
|-----------|------|---|-----|-------|
| **ds005555** | Bitbrain BOAS | 128 nights | **18–82 continuous** ✓ | **TOP sleep add.** Healthy adults, full-night PSG + wearable. The healthy reference NSRR structurally lacks |
| ds003768 | Simultaneous EEG-fMRI sleep | 33 | unconfirmed ⚠ | multimodal/niche |
| ANPHY-Sleep | (Scientific Data, **not OpenNeuro**) | 29 | 32±6 | **83-ch high-density** overnight; verify host before counting |

### MEG resting beyond Cam-CAN (both off-OpenNeuro — pull from native homes)

| Dataset | N | Home | Notes |
|---------|---|------|-------|
| OMEGA | ~220 | omega.bic.mni.mcgill.ca (only 5-subj sample = ds000247 on OpenNeuro) | resting MEG + age |
| MOUS | 204 | Donders Repository | 5-min eyes-open resting MEG + language |

### Ranked recommendation
1. **ds005385** (resting) and **ds005555 BOAS** (sleep) — both open, no wait, download now.
2. ds003775 as a second healthy resting set.
3. OMEGA / MOUS from native repos if/when a MEG age arm is built.

### NEMAR capability we're under-using
NEMAR links the **Neuroscience Gateway (NSG)** → **no-cost HPC on SDSC Expanse** (EEGLAB via
NSGportal, Python/MATLAB/R). Could run coffeine/REVE preprocessing **on-platform against
dataset pointers** instead of staging TBs to `/data`. Also pre-computes per-dataset QC stats
and supports HED event-tag search to extract eyes-closed segments from task datasets.

### Download
```
sbatch --export=ALL,DS=ds005385 download_openneuro_dgx.sbatch   # Dortmund Vital resting
sbatch --export=ALL,DS=ds005555 download_openneuro_dgx.sbatch   # BOAS healthy sleep
```
→ lands in `/data/datasets/openneuro/<ds>/`. Public S3, no credentials.

## OpenNeuro/NEMAR Sources

- OpenNeuro GraphQL API + S3 mirror `s3://openneuro.org/<ds>`
- NEMAR: https://nemar.org/about ; Neuroscience Gateway: https://www.nsgportal.org
- ds005385 (Dortmund Vital): https://www.nature.com/articles/s41597-024-03797-w
