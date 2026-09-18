# NEBULA X PS3 app

This directory is self-contained apart from the organiser input files: it includes the `nebulax`
package, four trained models, the production web bundle, prediction scripts and the small CV
result files displayed by the page. It is under 50 MB and does not need network access at run
time.

Use Python 3.11 with the app dependencies installed in the active environment. From this directory:

```bash
python -m pip install -r requirements.lock.txt
./run.sh 8765
```

Open `http://127.0.0.1:8765/`. An optional first argument selects another port. `run.sh` changes
to its own directory before starting Uvicorn, so it works regardless of the caller's current
directory. It uses the active Python, or a local Conda environment named `nebulax` if the active
Python lacks app dependencies. `NEBULAX_PYTHON` can select another interpreter. Stop it with Ctrl-C.

CLI prediction uses the same code path:

```bash
python predict.py --input /path/to/one/Test/file-or-folder --output predictions.csv
```

The raw PS3 datasets are not redistributed. Supply the released Test files through the page or
CLI. Training and ladder scripts are included for review, but refitting requires the organiser
training folders at the layout described in the main write-up.

The Predict page shows the four scored PS3 systems by default. Its **ALL SIX** filter also shows
brake air supply and axle bearing with compact, bundled MetroPT-3 and Ottawa recording excerpts.
The excerpts animate the corresponding component on the 3D train and can be downloaded as CSV.
They are research demonstrations and do not produce PS3 prediction CSVs. ACV `.xlsx` uploads
receive a quick workbook-format check before entering the queue; the full loader checks them again
when RUN is pressed.

If a browser's file picker does not return files, paste an absolute file path or Test-folder path
into **LOCAL FILE OR TEST FOLDER PATH** and press **ADD PATH**. This option is available only
through `run.sh` on the same computer; each file is still checked and predicted by the same model.

The earlier replay console remains available by direct link at `/?page=twin`. The bundled
two-train bearing and brake replay covers 12–14 September 2026 UTC; both systems accept
simulator fault injection. Door, ACV, Rail and SHM use labelled
model-predicted examples when the relevant local files are available. The Predict page's optional
replay frames remain in browser memory and do not survive a page reload.
