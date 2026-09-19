// Plain-language definitions for the explanation panel's (i) icons: one entry per key the API
// puts in `explanation.numbers` (nebulax/ps3/{door,acv,rail,shm}.py and stream.py) and per
// organiser CSV column. A key with no entry simply gets no icon - nothing is invented.

const COMMON = {
  start_time: "First sample of the door cycle, in the stream's own timestamp format (kept to the millisecond).",
  end_time: "Last sample of the door cycle; cycles never overlap.",
  file_id: "The Test file this row answers for, exactly as it was named in the upload.",
  confidence: "The model's abnormal probability for the cycle, attached when the API computed one for every row.",
};

const DEFS = {
  door: {
    ...COMMON,
    prediction: "Normal, or Abnormal resistance: the door drew more current than a normal door in the middle of its travel.",
    i_mid_rel: "Middle-travel current: the mean motor current over the cruise phase (15 to 85 % of travel) divided by the median of normal training cycles in the same direction. 1.0 is a typical door; the single strongest feature.",
    i_mid_ma: "Middle-travel current in milliamps, before normalisation.",
    threshold: "The saved probability cut-off for this direction's logistic model, tuned on inner contiguous splits during training. p above it is Abnormal.",
    p_abnormal: "The logistic model's probability that this cycle shows abnormal resistance.",
    duration_s: "Length of the cycle in seconds, from the first to the last sample.",
    n_segments: "Door cycles the segmenter cut from the stream.",
    n_abnormal: "Cycles labelled Abnormal resistance.",
    abnormal_rate: "Share of cycles labelled Abnormal resistance.",
    max_i_mid_rel: "The largest middle-travel current ratio seen in this stream.",
  },
  acv: {
    ...COMMON,
    ranked_cars: "Every car of the train-day, most to least likely to be leaking refrigerant, separated by |.",
    top_car: "The car ranked first: the largest mean peer-delta over hot, cooling-mode rows.",
    top_car_peer_delta_hot_k: "For the top car: its indoor temperature minus the median of the other cars at the same timestamp, averaged over hot rows, in kelvin.",
    top_car_cooling_fraction: "Share of the top car's usable rows in which it was in cooling mode.",
    rank_margin_to_2nd: "Score of the top car minus the score of the second car, in kelvin: how clear the call is.",
    n_unmapped_parameters: "Workbook columns the schema-aware loader could not map to a known field. They are reported, never guessed.",
    _pattern: [
      [/^car_(\d+)_peer_delta_hot_k$/, (m) => `Car ${m[1]}: its indoor temperature minus the median of the other cars at the same timestamp, averaged over hot rows (outdoor temperature at or above the case median), in kelvin. The ranking sorts this number.`],
    ],
  },
  rail: {
    ...COMMON,
    prediction: "Normal, Side I or Side II: which rail, if either, shows corrugation in this one-second run.",
    speed_kmh: "Train speed during the run. Below 20 km/h the run is called Normal outright: corrugation vibration cannot be told from noise at that speed.",
    side_max_rms_ratio_db: "The loudest axle box on Side I against the loudest on Side II, in decibels. Positive means Side I vibrates more.",
    side_i_minus_ii_db: "Same-side coherence contrast: how much more the Side I boxes agree with each other than the Side II boxes, in decibels. Corrugation excites a whole side together.",
    p_normal: "Averaged probability of Normal from the three seeds, after mirror test-time augmentation.",
    p_side_i: "Averaged probability of Side I corrugation.",
    p_side_ii: "Averaged probability of Side II corrugation, after the 1.25 class prior.",
    speed_kmh_so_far: "Mean speed over the samples streamed so far.",
    speed_ready: "1 once enough samples have arrived to trust the speed estimate.",
  },
  shm: {
    ...COMMON,
    prediction: "Cumulative fatigue damage of the record on the Miner scale: 1.0 is the end of life.",
    damage: "Cumulative fatigue damage predicted for this record (Miner's rule scale, 1.0 = end of life).",
    p2p: "Peak-to-peak stress range of the whole record.",
    blockmax_p2p_20: "Mean of the peak-to-peak range over 20 equal blocks: how large the swings are on average, not just the single worst one.",
    rainflow_cycles: "Number of stress cycles the rainflow counter extracted from the record.",
    cv_mape: "The model's cross-validated mean absolute percentage error; the band below comes from it.",
    damage_low: "Prediction minus its cross-validated error band.",
    damage_high: "Prediction plus its cross-validated error band.",
    life_fraction_used_pct: "The damage number as a percentage of the fatigue life.",
    damage_running: "Damage accumulated over the samples streamed so far.",
    damage_final: "Damage predicted for the whole record.",
    cycles_so_far: "Rainflow cycles counted over the samples streamed so far.",
  },
};

const VIEWPORT = {
  car: "Car of the 8-car train the explanation points the 3D model at (1 is the front cab).",
  side: "Side of the train, L or R, seen from the front cab; rail Side I / II map to it.",
  component: "The mesh the 3D model highlights for this row.",
};

/** Definition text for one key of one task, or null when there is none. */
export function defineNumber(task, key) {
  const table = DEFS[task] || COMMON;
  if (table[key]) return table[key];
  if (VIEWPORT[key]) return VIEWPORT[key];
  for (const [re, fn] of table._pattern || []) {
    const m = re.exec(String(key));
    if (m) return fn(m);
  }
  return null;
}
