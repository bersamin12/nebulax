# PS3 test fixtures

Tiny verbatim slices of the organisers' Problem Statement 3 datasets, cut on 18 Sep 2026 from
`readingmaterials/problem_statement/PS3/02_Datasets/` (that clone is gitignored; these slices are
committed so the PS3 tests run on a fresh checkout with no data download).

Every file is kept under the 200 KB fixture cap in `w4_common.md`. They are **schema and
round-trip fixtures**, not modelling data: 100 rows of a 10 kHz rail file is 10 ms of signal, so
feature code that needs a full second must synthesise its own input or read the real dataset.

| Path | Source | How it was cut |
|---|---|---|
| `door/train_slice.csv` | `Door/Train.csv` | Header + data rows 1-466 verbatim, i.e. exactly the first three labelled cycles (186 + 143 + 137 rows, back to back with no gap rows between them). All 17 columns. |
| `door/train_slice_answer.csv` | `Door/Train_Segments_Answer.csv` | Header + the matching first three answer rows (`train_seg_001`-`003`: Normal, Normal, Abnormal resistance). Covers both labels and both operations (Close, Open). |
| `rail/train1_slice.csv` | `Rail_Corrugation/Train/Train1.csv` (label `Normal`) | Header + the first 100 data rows verbatim, all 129 columns (rotating speed + 64 axle boxes x vibration/shock). The brief's 500 rows would have been ~880 KB, over the fixture cap; 100 rows is 178 KB. |
| `shm/train01_slice.csv` | `SHM/Train/train01.csv` (damage 0.103662995) | The first 5,000 lines verbatim. These files are headerless, single-column dynamic stress, CRLF line endings - the slice keeps all three properties. |
| `acv/case01_slice.xlsx` | `ACV/Train/acv_case_01.xlsx` (faulty car `01`) | First 30 rows, all 67 columns (3 identifying + 8 cars x 8 parameters), re-written with `pandas.to_excel(index=False)`. The 8-parameter header shape, which 5 of the 6 train cases and the test case use. |
| `acv/case04_wide_slice.xlsx` | `ACV/Train/acv_case_04.xlsx` (faulty car `01`) | First 8 rows, all 483 columns (3 + 8 cars x 60 parameters), same re-write. The odd case: the 60-parameter header shape, and cars 05-08 are all NaN in the full file. |

Re-cutting them (from the repo root, with the organisers' clone in place):

```bash
D=readingmaterials/problem_statement/PS3/02_Datasets
( head -1 $D/Door/Train.csv; sed -n '2,467p' $D/Door/Train.csv ) > tests/fixtures/ps3/door/train_slice.csv
( head -1 $D/Door/Train_Segments_Answer.csv; sed -n '2,4p' $D/Door/Train_Segments_Answer.csv ) > tests/fixtures/ps3/door/train_slice_answer.csv
head -101  $D/Rail_Corrugation/Train/Train1.csv > tests/fixtures/ps3/rail/train1_slice.csv
head -5000 $D/SHM/Train/train01.csv            > tests/fixtures/ps3/shm/train01_slice.csv
python -c "import pandas as pd; D='$D/ACV/Train/'; \
pd.read_excel(D+'acv_case_01.xlsx', nrows=30).to_excel('tests/fixtures/ps3/acv/case01_slice.xlsx', index=False); \
pd.read_excel(D+'acv_case_04.xlsx', nrows=8).to_excel('tests/fixtures/ps3/acv/case04_wide_slice.xlsx', index=False)"
```

Labels for the slices (from the organisers' label files) are in the table above; no label file is
copied here, so a test that needs one states it inline.
