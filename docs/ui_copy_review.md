# UI copy review

Status: approved and implemented on 19 September 2026. “Train Digital Twin” remains the product name.

Scope: the two user-facing pages—Overview and the prediction workflow—plus the shared header, tutorial, and relevant system messages. The hidden fleet replay dashboard is covered separately at the end.

## Recommended direction

The interface should sound like a technical tool, not a pitch deck or a generated research summary. The proposed voice is:

- neutral, direct, and specific;
- task-first: say what the user can do before explaining how it works;
- confident without self-certifying words such as “honest”, “real”, “nothing invented”, or “shipped”;
- plain English around the controls, with exact technical terms where they carry meaning;
- sentence case for prose and short labels; uppercase can remain as a visual style for compact navigation and panel labels;
- “the model” or “the app” instead of “we” wherever the team is not the subject.

Keep exact dataset names, model names, metrics, scores, class labels, CSV field names, and competition terminology unchanged. They are data, not brand copy.

## Product naming and shared header

The second page is primarily a prediction workflow with a 3D train view. Calling the whole page “Digital Twin” overstates what it does and makes the information architecture less clear. The recommended navigation label is **Predict**. “Train model” or “3D train view” can describe the visual itself.

| Location | Current | Proposed | Reason |
|---|---|---|---|
| Header product name | `TEAM BUS MRT WALK · TRAIN DIGITAL TWIN` | `TEAM BUS MRT WALK · TRAIN DIGITAL TWIN` | Preserves the approved product name. |
| Header subtitle | `TRACK 3 · PROBLEM STATEMENT 3 · CONDITION MONITORING OF ROLLING STOCK` | `TRACK 3 · PROBLEM STATEMENT 3` | The product name already supplies the context. |
| Main navigation | `OVERVIEW` / `DIGITAL TWIN` | `OVERVIEW` / `PREDICT` | Makes the destination and action clear. |
| Overview primary action | `OPEN DIGITAL TWIN` | `OPEN PREDICTION WORKSPACE` | Sets the right expectation. |
| Overview secondary action | `START TUTORIAL` | `TAKE A TOUR` | Shorter and more natural. |
| Static status chip | `OFFLINE · 4 / 4 MODELS` | `4 PREDICTION MODELS` | “Offline” can be confused with disconnected or unavailable. If model readiness is not checked here, avoid claiming it. |
| Tutorial button tooltip | `Guided tour of the Digital Twin page` | `Tour the prediction workflow` | Accurate and concise. |
| Mobile notice | `Limited support on phones and tablets. This is a desktop console; pinch to zoom, or open the link on a PC for the full experience.` | `This workspace is designed for desktop. On a phone or tablet, pinch to zoom or open it on a larger screen.` | Less defensive and avoids “full experience”. |

## Overview page

### Opening section

| Location | Current | Proposed | Reason |
|---|---|---|---|
| Hero eyebrow | `Team Bus MRT Walk · Problem Statement 3` | Keep | Useful context. |
| Hero heading | `A digital twin for a metro train's condition monitoring` | `Train condition monitoring, in one workspace` | Clearer, quieter, and aligned with what the app actually provides. |
| Hero paragraph | `The Train Digital Twin is a virtual copy of the train that runs the same four fault-detection models we submit: doors, air-conditioning, rail corrugation and structural fatigue. Upload a released Test file, press RUN, and watch the component light up on the train model with an explanation for every row.` | `Run the four Problem Statement 3 models for doors, air-conditioning, rail corrugation, and structural fatigue. Upload a Test file to review the predictions, inspect the affected component on the train, and download the submission CSV.` | Leads with the workflow; removes the promotional “watch it light up” phrasing. |
| Primary button | `Open Digital Twin` | `Open prediction workspace` | Matches the proposed navigation. |
| Tour button | `Take the 2-minute tour` | `Take a quick tour` | Avoids a time promise and sounds less templated. |
| Supporting line | `Runs offline · app and predict.py call the same functions` | `Runs locally using the same prediction code as predict.py` | More precise and easier to read. |

### Results and workflow

| Location | Current | Proposed | Reason |
|---|---|---|---|
| Score section label | `Honest headlines · nested / outer CV on training data only` | `Cross-validation results · training data only` | The validation method establishes credibility; “honest” does not. |
| Score source note | `Every number names its JSON key in results/ps3/leaderboard.md` | `Full results and source fields: results/ps3/leaderboard.md` | Shorter and more neutral. |
| Workflow label | `How the digital twin works` | `Prediction workflow` | Direct and accurate. |
| Workflow heading | `Four steps from a raw Test file to a prediction on the train` | `From Test file to prediction in four steps` | Removes filler. |
| Workflow paragraph | `The same four functions run whether you press RUN in the browser or call python predict.py --input …. The table on screen is the CSV you download, and the train model lights up the component behind the selected row.` | `The browser and predict.py use the same prediction functions. Results appear in submission format, and selecting a row highlights its component on the train.` | Keeps the useful assurance without sounding defensive. |
| Step 1 heading | `Pick a subsystem, queue files` | `Choose a system and add files` | “Add” is more familiar than “queue” at the start of the flow. |
| Step 1 body | `One Door stream, one ACV workbook, 68 Rail files or 16 SHM files. The browser sends at most 32 files per request against one session, so the table fills as batches return.` | `Add a Door CSV, ACV workbook, or a folder of Rail or SHM files. Large selections are processed in batches, and results appear as each batch finishes.` | Moves implementation details out of the main explanation. |
| Step 2 heading | `Schema-aware loader` | `Check the file format` | Describes the user-visible outcome. |
| Step 2 body | `Parses timestamps to the millisecond, discovers Car N - parameter columns by pattern, maps header aliases to canonical fields and reports unknown columns instead of guessing.` | `The app checks timestamps and column names, recognises known aliases, and flags fields it cannot use.` | Easier to scan; preserves the behaviour. |
| Step 3 heading | `Physics features` | `Build model inputs` | More inclusive of the different feature types. |
| Step 4 heading | `Saved artifact, frozen` | `Run the selected model` | User-facing rather than implementation-facing. |
| Component section label | `Where each subsystem lives · one cab, cycling every 4 seconds` | `Subsystem locations on the train` | Removes animation implementation from the heading. |
| Component note | `The component behind the selected row lights up green (normal) or red (fault)` | `Selecting a result highlights the corresponding component: green for normal and red for a detected fault.` | Neutral and explicit. |

### Technical and research sections

These sections should remain technical, but their headings and framing can be calmer.

| Location | Current | Proposed | Reason |
|---|---|---|---|
| Model section label | `Main models` | `Prediction models` | More specific. |
| Model section heading | `One model per subsystem, chosen on a frozen ladder` | `One model for each scored system` | “Frozen ladder” is internal language. |
| Model intro | `Each page says what the model eats, how it decides, and which studies pointed us to it; the ablation section further down shows what the larger models scored. None of the four deployed models is a neural network: every artifact is under 1 MB and predicts in seconds on a laptop, which is what an offline twin needs.` | `Each tab summarises the model input, decision method, validation result, and supporting research. All four models are under 1 MB and run locally on a laptop; larger alternatives are compared in the ablation section.` | Removes metaphor, repetition, and self-justifying tone. |
| Repeated callout label | `IN PLAIN TERMS` | `SUMMARY` | Less patronising and more neutral. |
| Repeated row label | `Why this` | `Selection rationale` | More precise. |
| Repeated source label | `Literature that pointed us here` | `Supporting research` | Removes the narrative template. |
| EDA heading | `What the released data look like, one track at a time` | `Released dataset profiles` | Shorter and more conventional. |
| Repeated EDA label | `What the EDA changed in the model` | `Design implications` | Clearer and less conversational. |
| Ablation heading | `What each design choice bought, one track at a time` | `Effect of each design choice` | Removes an AI-like metaphor. |
| Ablation intro | `An ablation removes or swaps one ingredient and re-runs the same validation, so the score difference is attributable to that ingredient. Each page groups the rows by what was varied, names the metric and split, and lists what was tried but not shipped.` | `Each ablation changes one part of the pipeline and repeats the same validation. The tables show the metric, split, alternatives tested, and the configuration selected for the final model.` | Removes recipe metaphors and “shipped”. |
| Repeated ladder title | `Door: what we ablated` and equivalents | `Door ablation results` and equivalents | Conventional research wording. |
| Repeated rejected-work label | `Tried, not promoted` | `Other configurations tested` | Neutral and easier to understand. |
| Repeated takeaway label | `What we learned` | `Key findings` | Less generated-sounding. |
| Research section label | `Exploratory · not scored by PS3` | `Research datasets · outside PS3 scoring` | Clearer relationship to the workflow. |
| Research heading | `Additional systems and datasets we brought into the twin` | `Additional condition-monitoring datasets` | Removes “we” and the unsupported implication that these are part of a full twin. |
| Research intro | `These are exploratory systems we found online using open public datasets, which we also include in our twin. Neither is scored by PS3. They appear on the Digital Twin page as read-only dataset profiles, they feed the research benchmark (285 retained runs, one seed each) and they drive the two physics simulators behind the FAULT / EXAMPLE controls.` | `These public datasets support exploratory work on axle bearings and brake-air supply. They are not part of PS3 scoring. In the prediction workspace, each appears as a read-only dataset profile with a short recorded example.` | Matches the two-page product and removes references to hidden replay controls. |
| Bearing “On the twin” row | `Click any axle box on the Digital Twin page: the detail panel shows the envelope features and the lumped thermal node for that box; FAULT injects a seeded defect` | `Open Axle bearing in the prediction workspace to view a recorded vibration excerpt and its location on the train. This view does not run a PS3 prediction.` | Describes the current user-facing behaviour. |
| Brake-air “On the twin” row | `Click the APU under Car 3: reservoir pressure, motor current and tower switching trace live; FAULT opens a leak and the alarm episode appears on the transport bar` | `Open Brake air supply in the prediction workspace to view a recorded compressor excerpt and its location beneath Car 3. This view does not run a PS3 prediction.` | Removes references to the hidden dashboard. |
| Repeated caveat label | `Honest notes` | `Limitations` | The content, not the label, should establish credibility. |
| Team heading | `Four final-year engineers from the Renaissance Engineering Programme` | `Team Bus MRT Walk` | Avoids repeating information already listed under every member. |
| Evidence label | `Evidence` | `References and results` | More specific. |

## Prediction workflow

### System selection and task descriptions

| Location | Current | Proposed | Reason |
|---|---|---|---|
| Systems summary | `4 models · 2 exploratory` | `4 prediction models · 2 research datasets` | “Exploratory” alone does not explain what the tiles are. |
| Research tile state | `(EXPLORATORY)` | `RESEARCH` | Shorter and less vague. |
| Research badge | `EXPLORATORY · NO UPLOAD OR PREDICTION` | `READ-ONLY DATASET · NO PREDICTION` | States the actual limitation. |
| Door title | `Door segment detection` | `Door cycle classification` | The output class is the main task; segmentation is part of the pipeline. |
| Door description | `Cut the motor-current stream into door cycles and label each Normal or Abnormal resistance.` | `Identify each door cycle in the motor-current stream and classify it as Normal or Abnormal resistance.` | Grammatically complete and less mechanical. |
| ACV title | `Refrigerant-leak car ranking` | Keep | Already clear. |
| ACV description | `Rank every car of a train-day workbook from most to least likely to be leaking.` | `Rank the cars in each workbook from most to least likely to have a refrigerant leak.` | More natural phrasing. |
| Rail title | `Rail corrugation side` | `Rail corrugation classification` | Names the task rather than a field. |
| Rail description | `Call each run Normal, Side I or Side II from the 64 axle-box accelerometers.` | `Classify each run as Normal, Side I, or Side II using the 64 axle-box accelerometers.` | Neutral verb and standard punctuation. |
| SHM title | `Fatigue-damage regression` | `Fatigue damage prediction` | User-facing rather than modelling jargon. |
| SHM description | `Predict the cumulative fatigue damage of a dynamic-stress record.` | `Estimate cumulative fatigue damage from a dynamic stress record.` | More natural and avoids repeating “predict”. |
| Research context intro | `Research and calibration data for the train component shown above. This view does not produce a PS3 prediction.` | `Background data for the component shown above. This read-only view does not run a PS3 model.` | Shorter and clearer. |
| Empty research note | `No uploaded-file prediction or submission CSV is available for this research system.` | `Uploads and submission CSVs are not available for this dataset.` | Plain, direct language. |

### Adding and checking files

| Location | Current | Proposed | Reason |
|---|---|---|---|
| Drop zone | `drop files or a folder here` | `Drop files or a folder here` | Sentence case and normal UI grammar. |
| File input tooltip | `Choose files to add to the queue` | `Choose files` | The surrounding UI already shows the queue. |
| Folder input tooltip | `Choose a folder to add its files to the queue` | `Choose a folder` | Same. |
| Path fallback label | `LOCAL FILE OR TEST FOLDER PATH · FIREFOX FALLBACK` | `LOCAL PATH · USE IF THE FILE PICKER DOES NOT WORK` | Explains when it is useful without naming a browser unnecessarily. |
| Empty queue | `nothing queued` | `No files added` | Familiar and neutral. |
| No-file picker error | `The picker returned no files. Try dropping a file into the box above.` | `No files were selected. Choose a file or drop it here.` | Plain language. |
| Successful check | `1 file checked and queued` | `1 file added` | The check is already shown in the confirmation step. Use the plural equivalent dynamically. |
| Partial check | `3 of 5 files matched and were queued; the rest were left out` | `3 of 5 files were added. The other 2 did not match the required format.` | More explicit and easier to understand. |
| Cancelled check | `pick cancelled, nothing was queued` | `File selection cancelled` | Concise; no need to narrate unchanged state. |
| Confirmation eyebrow | `CHECK THE FILES` | `REVIEW FILES` | Less command-like. |
| Confirmation question | `Is this the right input for …?` | `Check this file before adding it` | The app has already evaluated the format; a question is unnecessary. Use `Check these files before adding them` for plural. |
| Expected-format label | `THIS MODEL EXPECTS` | `REQUIRED FORMAT` | Shorter and clearer. |
| Valid verdict | `MATCHES` | `READY` | Describes the next state. |
| Invalid verdict | `DOES NOT MATCH` / `NO MATCH` | `CHECK FORMAT` | More useful and consistent. |
| Confirm action | `QUEUE 1 FILE` | `ADD 1 FILE` | “Add” is more natural for file selection. Use `ADD THE 3 READY FILES` for a partial match. |
| Invalid confirm action | `NOTHING TO QUEUE` | `NO VALID FILES` | Describes why the action is disabled. |
| Re-check note | `the server re-checks every file on upload` | `Files are checked again when processing starts.` | Complete sentence and clearer timing. |

### Running predictions and reviewing results

| Location | Current | Proposed | Reason |
|---|---|---|---|
| Replay checkbox | `INCLUDE REPLAY FRAMES (stream frames)` | `GENERATE 3D PREVIEW` | Says what the option provides. If the hidden dashboard is removed and nothing consumes these frames, remove the option instead. |
| Idle helper | `queue files to run` | `Add files to begin` | Direct and familiar. |
| Run button | `RUN 3` | `RUN 3 FILES` | The number has context. |
| Running state | `RUNNING 2/3` | `PROCESSING 2 OF 3` | Clearer status language. |
| Stop tooltip | `Stop after the file in flight; the rows already predicted stay, the rest of the queue waits` | `Stop after the current file. Completed results will remain, and unprocessed files will stay in the queue.` | Complete, readable sentences. |
| Empty results | `no predictions yet: queue files on the left and press RUN` | `No predictions yet. Add files on the left, then select Run.` | Friendlier without becoming chatty. |
| Running empty state | `predicting…` | `Processing files…` | Describes the operation. |
| Default results guidance | `1. Pick a system on the left. 2. Choose or drop its Test files and confirm the check. 3. Press RUN. One row per prediction lands here; click it to see why.` | `Choose a system, add its Test files, and select Run. Results will appear here; select a row to inspect its details.` | Removes duplicated instructions and “lands here”. |
| Table hint | `CLICK A ROW TO INSPECT IT` | `SELECT A ROW FOR DETAILS` | More conventional interface language. |
| Row tooltip | `Click to select this row: the train model and the explanation follow` | `Select this row to update the train view and details.` | Avoids “follow”, which sounds generated. |
| Explanation empty state | `run a prediction, then pick a row` | `Run a prediction, then select a result.` | Consistent terminology. |
| ACV rank detail | `most likely leak` | `highest leak likelihood` | More precise. |
| Missing trace | `no trace` | `No trace available` | Complete empty state. |
| Missing explanation | `this task sent no explanation payload for the row` | `No explanation is available for this result.` | Hides API implementation language. |
| Download button | `DOWNLOAD … CSV` | Keep | Direct and useful. |
| Clear button | `CLEAR` | `CLEAR SESSION` | Makes the scope of the action explicit. |
| Scoring label | `HOW IT'S SCORED` | `SCORING` | Compact and neutral. |
| No-session footer | `no session yet: the CSV link appears after the first successful batch` | `The CSV will be available after the first successful result.` | User outcome instead of implementation state. |
| Active-session footer | `session 12345678… · 3 file(s) · 18 row(s) · the download is the validated submission CSV` | `Session 12345678… · 3 files · 18 results · CSV ready` | Shorter; use correct singular/plural forms. |
| Server banner | `SERVER UNREACHABLE` | `PREDICTION SERVICE UNAVAILABLE` | Names the affected capability. |
| Server banner detail | `The page still works, but nothing can be predicted until /api/ps3 answers.` | `You can review the page, but predictions are unavailable until the service reconnects.` | Removes endpoint jargon from the UI. |

### Tutorial

The tutorial is useful, but several steps defend the implementation rather than help the user. The revised version should be shorter enough to scan while the highlighted controls remain visible.

| Step | Current title | Proposed title | Proposed body |
|---|---|---|---|
| Welcome | `Welcome to the Train Digital Twin` | `Prediction workspace` | `Choose a system, add Test files, run a model, and inspect the results on the train. This short tour covers the workflow from left to right.` |
| Systems | `Pick a subsystem` | `Choose a system` | `The four scored systems run prediction models. Brake air supply and Axle bearing are read-only research datasets. Select a tile to view its description.` |
| Files | `Queue the Test files` | `Add Test files` | `Choose files, select a folder, or drag and drop. The app checks the format before adding valid files to the queue.` |
| Run | `Press RUN` | `Run the model` | `Select Run to process the queue. Results appear as files finish. Turn on 3D Preview if you want preview frames to be generated.` |
| Results | `Read the predictions` | `Review results` | `Select a row to update the train view and the details panel. For ACV, select a car to inspect its rank.` |
| Train | `The train lights up` | `Inspect the train view` | `The selected component is highlighted by status. Drag to rotate, scroll to zoom, or select a component directly.` |
| Explanation | `Why this prediction` | `Review the details` | `This panel shows the values, trace, and train location returned for the selected result. Use the information icons for variable definitions.` |
| Download | `Download the submission CSV` | `Download or start again` | `Download the validated submission CSV, or clear the session to begin another run. You can restart this tour from the header.` |

Toast proposal: replace `Tour finished. Restart it any time from TUTORIAL in the header.` with `Tour complete. You can restart it from the header.`

## Wording to standardise across both pages

| Avoid | Use instead |
|---|---|
| `Digital Twin` for the whole prediction page | `prediction workspace`; reserve `train view` for the 3D visual |
| `honest`, `kept honest` | name the actual method or limitation |
| `shipped`, `deployed` | `selected`, `final`, or `current` |
| `what the model eats` | `model input` |
| `lights up` | `highlights` |
| `pick` in visible UI | `choose` or `select` |
| `queue` before a file has been added | `add`; keep `queue` for the actual queue panel |
| `nothing is invented` | remove; state the source of the displayed data if needed |
| API terms such as `payload` or endpoint paths | describe the user-visible state |
| sentence fragments in messages | short complete sentences |

## Hidden fleet dashboard

There are currently three routes in `App.jsx`, but only two appear in the navigation:

1. `overview` — user-facing;
2. `predict` — user-facing;
3. `twin` — legacy replay dashboard, still reachable through `?page=twin` and old train/component deep links.

The replay dashboard has been archived separately from the active application. The shared 3D viewport remains in the prediction page.

Implemented structure:

1. Remove `twin` from active page routing and stop redirecting old train/component query parameters to it.
2. Keep shared prediction components in place:
   - `components/viewport/TrainViewport.jsx`
   - `components/viewport/componentMap.js`
   - `components/viewport/useHealthMaterials.js`
   - the viewport CSS, elevation fallback, and 3D assets
3. Move dashboard-only code under `web/src/archive/fleet-dashboard/`:
   - the `TwinConsole` currently embedded in `App.jsx`;
   - `AlertTicker.jsx`;
   - `SubsystemLanes.jsx`;
   - `LayersPanel.jsx`;
   - `DetailPanel.jsx`;
   - `components/transport/`;
   - `state/replayStore.js`;
   - the transport and viewport development entry points where they exist only to support the dashboard.
4. Audit `state/ps3StreamStore.js` before moving it. The prediction hook currently writes preview-frame results to this store, while the legacy dashboard is the main consumer. If the dashboard route is removed, the `INCLUDE REPLAY FRAMES` option and this write path may be removable too.
5. Put a short README in the archive folder explaining the former `?page=twin` route, its dependencies, and the commit from which it was archived.

Compatibility behaviour: old `?page=twin`, `?train=…`, or `?component=…` links open the prediction workspace. Query parameters that no longer apply are removed from the URL.

## Suggested implementation order

1. Keep **Train Digital Twin** as the product name and use **Predict** as the workflow navigation label.
2. Apply the shared-header, hero, prediction-workflow, empty-state, and tutorial edits first. These have the largest effect on tone.
3. Apply the heading and framing edits across the long technical Overview sections without changing their measurements or conclusions.
4. Archive the replay dashboard in a separate commit, then remove any prediction options that only fed that dashboard.
5. Run a final visual pass for wrapping. Several proposed labels are shorter, but the new header product name and `OPEN PREDICTION WORKSPACE` should still be checked at the fixed 1440 × 900 layout and minimum desktop scale.
