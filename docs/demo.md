# PS3 demo run sheet — target 2:45, hard stop 3:00

## Before recording

From the repository root, with the `nebulax` environment active:

```bash
(cd web && npx vite build)
uvicorn nebulax.api.main:app --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/?page=predict&task=door` at 1440×900. Close other tabs and clear any
old queue. The built app is `web/dist`; no Vite development server is needed. Network may be
disabled: prediction, explanations, the 3D asset and CSV downloads are local. Do not open Fleet
twin during this PS3 recording.

The 1440×900 headless-Chrome rehearsal on the packaged models measured Door **3.23 s**, ACV
**5.24 s**, all 68 Rail files **29.59 s**, and all 16 SHM files **6.04 s** (44.10 s compute in
total). The timeline below therefore leaves more than two minutes for narration and interaction.

## Timed script

**0:00–0:12 — frame the app.** “NEBULA X covers all four PS3 tasks with one offline app. The
centre table is exactly what we submit; the right panel explains the selected row.” Point to the
four tabs and the validation metric printed on each.

**0:12–0:42 — Door.** On Door, choose
`readingmaterials/problem_statement/PS3/02_Datasets/Door/Test.csv`, press **RUN 1**, and say:
“This is one continuous 50 Hz stream. We segment it end to end and retain millisecond source
timestamps.” Show **38 rows**. Select an `Abnormal resistance` row so the door is red; point to
middle-travel current/profile numbers and the bounded trace. Press **DOWNLOAD
door_predictions.csv**.

**0:42–1:04 — ACV.** Press **CLEAR**, select ACV, choose
`readingmaterials/problem_statement/PS3/02_Datasets/ACV/Test/acv_test_case.xlsx`, then **RUN 1**.
Read the ranking from the table: `01|03|04|08|07|06|02|05`. Say: “The top car is highlighted;
all eight IDs come from this workbook's own headers. The rule compares each car with its seven
same-train peers and puts missing cars last.” Download `acv_predictions.csv`.

**1:04–1:42 — Rail.** Press **CLEAR**, select Rail Corrugation, then **CHOOSE FOLDER** and choose
`readingmaterials/problem_statement/PS3/02_Datasets/Rail_Corrugation/Test/`. Press **RUN 68**.
Say while it runs: “The browser sends at most 32 files per request; the server predicts one file
at a time and accumulates one validated CSV.” Show the queue reach **68 / 68**, select one Side
I/II result if present, and point to speed and the wavelength/spectral explanation. Download
`rail_predictions.csv`.

**1:42–2:05 — SHM.** Press **CLEAR**, select SHM, choose the folder
`readingmaterials/problem_statement/PS3/02_Datasets/SHM/Test/`, and press **RUN 16**. Select a
row and show the positive damage estimate plus rainflow/amplitude trace. Download
`shm_predictions.csv`.

**2:05–2:25 — submission integrity.** Say: “These downloads and `python predict.py` use the same
loader, model and CSV renderer. The delivered `predictions.zip` contains these four root-level
CSVs and is validated against the organiser schemas and Test file IDs.”

**2:25–2:45 — evidence.** Open `results/ps3/leaderboard.md` briefly. Say: “Every ladder number
names its JSON key. The honest headlines are Door 0.9818 ± 0.0364, exploratory ACV 0.9792 ±
0.0510, Rail 0.7571 ± 0.1230, and SHM score 0.9797. Every scaler, threshold, template, feature
choice and augmentation is fitted inside the training fold; held-out files are never augmented.”

Stop by **2:45**. The remaining 15 seconds are recording margin, not extra content.

## Six-system twin walkthrough

This walkthrough demonstrates the unified six-system fleet twin, bringing offline Problem Statement 3 uploaded predictions together with the real-time simulation digital twin:

1. **Predict page uploads:**
   - Start at `http://127.0.0.1:8765/?page=predict&task=door`.
   - **Door:** Choose `readingmaterials/problem_statement/PS3/02_Datasets/Door/Test.csv` and press **RUN 1** (~4 s).
   - **ACV:** Switch to ACV tab, choose `readingmaterials/problem_statement/PS3/02_Datasets/ACV/Test/acv_test_case.xlsx`, and press **RUN 1** (~22 s).
   - **Rail:** Switch to Rail corrugation tab, choose 3 test files (`Test1.csv`, `Test2.csv`, `Test3.csv` from `readingmaterials/problem_statement/PS3/02_Datasets/Rail_Corrugation/Test/`), and press **RUN 3** (~3–4 s).
   - **SHM:** Switch to SHM tab, choose 3 test files (`test01.csv`, `test02.csv`, `test03.csv` from `readingmaterials/problem_statement/PS3/02_Datasets/SHM/Test/`), and press **RUN 3** (~15 s).
   *(Note: The upload and streaming inference of 3 Rail + 3 SHM files takes ~19–40 s on this box; state this timing to the audience.)*

2. **Open the direct-link replay console separately:**
   - Open `http://127.0.0.1:8765/?page=twin`. The Predict page keeps the replay console out of its visible navigation. A fresh load shows simulator and locally available examples; browser-only replay frames from a Predict upload do not survive a page reload.
   - In the **SIX-SYSTEM LAYERS** panel on the left, click **ALL ON** to activate all six subsystems:
     1. Brake air supply (`sim · MetroPT-3 calibrated`)
     2. Axle bearing (`sim · Ottawa calibrated`)
     3. Door (`LTA PS3`)
     4. ACV refrigerant leak (`LTA PS3 upload`)
     5. Rail corrugation (`LTA PS3 upload`)
     6. SHM fatigue damage (`LTA PS3 upload`)
   - Open **FAULT / EXAMPLE**. Brake and bearing use simulator injection. Door, ACV, Rail and SHM replay a clearly labelled model-predicted LTA Test example, then hold its selected frame; **STOP** restores the previous uploaded stream and cursor.

3. **Live Playback (30 s):**
   - Let the twin play for 30 seconds. The unified transport bar drives all six synchronized clocks:
     - Door plays back-to-back over 30 s.
     - ACV, Rail, and SHM stream each file over 6 s at 1x speed, looping automatically.
   - The presenter reads out the live captions under the viewport, particularly the **honesty lines**:
     - **DOOR:** `DOOR · cycle 10/38 · causal preview · final frame = submitted row`
     - **RAIL:** `RAIL · Test2.csv · 0.3 s · estimating speed · demo ordering (files are unordered)` until tachometer transitions arrive.
     - **SHM:** `SHM · test02.csv · 581120 sample · D = 0.101 → 0.807 · Miner accumulation, ends on submitted value · demo ordering (files are unordered)`
     - **ACV:** `ACV · 20.7 h · car 04 rank 1 · prefix ranking`
     *(Emphasize the explicit "demo ordering (files are unordered)" disclaimers on Rail and SHM, the causal cycle guarantee on Door, and the peer-normalized prefix ranking on ACV.)*

4. **Component Inspection:**
   - Click one mesh per PS3 layer on the side-elevation schematic:
     - **Door:** Click Car 1 Door L1 — shows current cycle metrics (`p_abnormal`), cycle progress, and submitted prediction.
     - **Rail:** Click Rail Side I/II — shows speed and roughness metrics; axle boxes belong to the bearing simulator.
     - **SHM:** Click Bogie 1 or 2 on Car 1 — shows Miner damage accumulation ($D_{\text{running}} \to D_{\text{final}}$).
     - **ACV:** Click Car 1–8 roof AC units — shows peer ranking and excursion scores.
   - The right-hand **Detail Panel** seamlessly transitions between PS3 uploaded stream metrics and simulator physics traces for sim components.

## CLI equivalents

```bash
python predict.py --input readingmaterials/problem_statement/PS3/02_Datasets/Door/Test.csv --output door_predictions.csv
python predict.py --input readingmaterials/problem_statement/PS3/02_Datasets/ACV/Test --output acv_predictions.csv
python predict.py --input readingmaterials/problem_statement/PS3/02_Datasets/Rail_Corrugation/Test --output rail_predictions.csv
python predict.py --input readingmaterials/problem_statement/PS3/02_Datasets/SHM/Test --output shm_predictions.csv
```
