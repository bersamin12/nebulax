"""Tests for ``nebulax.advisory`` (app contract section 6). No network, no GPU, fast."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from nebulax.advisory import (
    CBM_STEPS,
    CLAUDE_MODEL,
    GLOSSARY,
    SYSTEM,
    TEMPLATE_MODEL,
    UNKNOWN_CONFIDENCE_CAP,
    Advisory,
    AdvisoryDraft,
    AlertContext,
    advise,
    build_user_message,
    classify,
    coerce_fault,
    find_api_key,
    parse_env_file,
    request_timeout,
    rule_keys,
    signal_aliases,
    template_advisory,
)
from nebulax.schema import FAULT_TYPES

T0 = datetime(2026, 9, 14, 8, 30, tzinfo=timezone.utc)


def make_alert(subsystem: str, signals, **kw) -> AlertContext:
    defaults = dict(
        episode_id=f"T01-{subsystem}-1",
        train_id="T01",
        car=1,
        subsystem=subsystem,
        component_id={"door": "door_L1", "pneumatic": "apu_1", "bearing": "axlebox_3R"}[
            subsystem
        ],
        model_name={"door": "cusum_cycle_scalar", "pneumatic": "sparse_autoencoder",
                    "bearing": "cusum_cycle_scalar"}[subsystem],
        score=5.9,
        threshold=2.4,
        t_start=T0,
        t_end=None,
        duration_h=6.2,
        top_signals=[{"signal": s, "z": z, "value": v} for s, z, v in signals],
        recent_events=["08:12 reversal"],
        fleet_peer_summary="fleet median well below this unit",
    )
    defaults.update(kw)
    return AlertContext(**defaults)


# --------------------------------------------------------------------------------------
# Template path: fault mapping per subsystem
# --------------------------------------------------------------------------------------


# The signal names below are spellings that really occur in ``data/scores/*.parquet`` -
# window statistics (``<channel>_<stat>``) and peer-normalised cycle features
# (``<feature>_peer_delta`` / ``_peer_z``). See OBSERVED_SIGNALS for the full vocabularies.
@pytest.mark.parametrize(
    "subsystem, signals, expected",
    [
        ("door", [("closing_time", 5.1, 3.9), ("i_mean_cruise", 3.2, 2.1)], "friction"),
        ("door", [("energy_J", 4.4, 410.0)], "friction"),
        ("door", [("pos_err_rms_peer_delta", 4.0, 0.02), ("closing_time", 3.0, 3.9)], "friction"),
        # backlash needs the start-current peak; position error alone is friction
        ("door", [("i_start_peak", 6.0, 9.1), ("pos_err_max", 3.0, 0.02)], "backlash"),
        ("door", [("pos_err_rms", 4.0, 0.01)], "friction"),
        ("door", [("dropout_frac", 7.0, 0.4)], "dcu_dropout"),
        ("door", [("i_peak", 3.3, 7.0)], "brush_wear"),
        ("door", [("shock_jump_count_est", 4.0, 5.0)], "shock_wear"),
        ("door", [("pos_close_max", -4.0, 0.9)], "misalignment"),
        (
            "pneumatic",
            [("Reservoirs_mean", -5.0, 7.2), ("Motor_current_mean", 3.0, 4.1)],
            "air_leak",
        ),
        ("pneumatic", [("H1_mean", -6.0, 0.0), ("TP2_mean", 4.0, 9.5)], "air_leak"),
        (
            "pneumatic",
            [("Oil_temperature_mean", 6.0, 92.0), ("Oil_temperature_max", 5.0, 98.0)],
            "oil_leak",
        ),
        ("pneumatic", [("Oil_temperature_rms", 4.0, 88.0)], "oil_leak"),
        ("pneumatic", [("TP2_mean", 5.5, 10.2), ("TP3_mean", -3.0, 8.1)], "clogged_filter"),
        (
            "pneumatic",
            [("Towers_mean", 4.0, 0.4), ("DV_pressure_std", 3.0, 0.2)],
            "dryer_valve_stuck",
        ),
        (
            "pneumatic",
            [
                ("Motor_current_mean", 4.0, 9.0),
                ("TP2_mean", -3.5, 8.1),
                ("Flowmeter_mean", -3.0, 20.0),
            ],
            "compressor_wear",
        ),
        ("pneumatic", [("Flowmeter_max", 5.0, 60.0)], "valve_leak_outlet"),
        ("bearing", [("vib_bpfo_mean", 5.0, 0.8)], "outer_race"),
        ("bearing", [("vib_kurt_mean", 4.0, 9.0)], "inner_race"),
        ("bearing", [("vib_rms_mean_peer_z", 4.0, 0.9)], "bearing_degradation"),
        ("bearing", [("T_box_std", -5.0, 0.0)], "sensor_stuck"),
        ("bearing", [("T_box_mean_peer_z", -6.0, 18.0), ("dT_peer_same_side_peer_z", -4.0, -3.0)],
         "sensor_offset"),
    ],
)
def test_template_fault_mapping(subsystem, signals, expected):
    alert = make_alert(subsystem, signals)
    fault, _reason = classify(alert)
    assert fault == expected
    adv = template_advisory(alert)
    assert adv.likely_fault == expected
    assert adv.likely_fault in FAULT_TYPES[subsystem]


# --------------------------------------------------------------------------------------
# Signal vocabulary: every rule key must resolve against what the scores actually carry
# --------------------------------------------------------------------------------------

#: The distinct ``top_signals_json`` column names observed in ``data/scores/scores.parquet``
#: and ``data/scores/metropt3.parquet`` (2026-09 demo scoring run), minus the 45
#: ``current_profile_50_*`` door bins, which are raw waveform samples and carry no rule.
#: Hard-coded on purpose: this test must not read ``data/``.
OBSERVED_SIGNALS = {
    "door": [
        "T_amb", "T_motor", "T_motor_peer_delta", "closing_time", "closing_time_peer_delta",
        "dropout_frac", "dropout_frac_peer_delta", "energy_J", "energy_J_peer_delta",
        "hour_of_day", "i_end", "i_end_peer_delta", "i_mean_cruise", "i_mean_cruise_peer_delta",
        "i_peak", "i_peak_peer_delta", "i_rms_cruise", "i_rms_cruise_peer_delta",
        "i_start_peak", "i_start_peak_peer_delta", "load_frac", "opening_time",
        "opening_time_peer_delta", "pos_close_max", "pos_close_max_peer_delta", "pos_err_max",
        "pos_err_max_peer_delta", "pos_err_rms", "pos_err_rms_peer_delta", "pos_open_max",
        "pos_open_max_peer_delta", "pwm_mean", "pwm_mean_peer_delta", "shock_jump_count_est",
        "shock_jump_count_est_peer_delta",
    ],
    "bearing": [
        "T_amb", "T_box_max", "T_box_max_peer_delta", "T_box_max_peer_z", "T_box_mean",
        "T_box_mean_peer_delta", "T_box_mean_peer_z", "T_box_min", "T_box_min_peer_delta",
        "T_box_min_peer_z", "T_box_std", "T_box_std_peer_delta", "T_box_std_peer_z", "axle",
        "axle_peer_delta", "axle_peer_z", "dT_opposite", "dT_opposite_peer_delta",
        "dT_opposite_peer_z", "dT_peer_same_side", "dT_peer_same_side_peer_delta",
        "dT_peer_same_side_peer_z", "dwell_fraction", "dwell_fraction_last_hour", "load_frac",
        "peer_median_same_side", "peer_median_same_side_peer_delta", "thermal_residual",
        "thermal_residual_peer_delta", "thermal_residual_peer_z", "v_mean", "vib_bpfo_mean",
        "vib_bpfo_mean_peer_delta", "vib_bpfo_mean_peer_z", "vib_crest_mean",
        "vib_crest_mean_peer_delta", "vib_crest_mean_peer_z", "vib_kurt_mean",
        "vib_kurt_mean_peer_delta", "vib_kurt_mean_peer_z", "vib_rms_max",
        "vib_rms_max_peer_delta", "vib_rms_max_peer_z", "vib_rms_mean",
        "vib_rms_mean_peer_delta", "vib_rms_mean_peer_z",
    ],
    # sim APU: window stats over the simulated channels (63 distinct names)
    "pneumatic": [
        f"{channel}_{stat}"
        for channel in (
            "DV_pressure", "Flowmeter", "H1", "Motor_current", "Oil_temperature",
            "Reservoirs", "TP2", "TP3", "Towers",
        )
        for stat in ("mean", "std", "min", "max", "slope", "rms", "kurtosis", "crest")
    ],
}

#: MetroPT-3 carries exactly these seven, and nothing else.
METROPT3_SIGNALS_SEEN = [
    "DV_pressure_mean", "H1_mean", "Motor_current_mean", "Oil_temperature_mean",
    "Reservoirs_mean", "TP2_mean", "TP3_mean",
]


def _resolvable(names):
    return {alias for name in names for alias in signal_aliases(name)}


@pytest.mark.parametrize("subsystem", sorted(OBSERVED_SIGNALS))
def test_every_rule_key_resolves_against_the_observed_signals(subsystem):
    """A rule keyed on a column the scores never emit is a dead rule (fix round 1)."""
    resolvable = _resolvable(OBSERVED_SIGNALS[subsystem])
    unreachable = sorted(key for key in rule_keys(subsystem) if key not in resolvable)
    assert unreachable == []


def test_metropt3_vocabulary_is_covered_by_the_pneumatic_rules():
    """Every one of the seven MetroPT-3 columns must be able to speak for some fault."""
    from nebulax.advisory import RULES

    keys = {key for rule in RULES["pneumatic"] for key in rule.signals}
    for name in METROPT3_SIGNALS_SEEN:
        assert keys & set(signal_aliases(name)), name


def test_signal_aliases_strips_peer_then_stat_and_keeps_compound_names():
    assert signal_aliases("T_box_max_peer_z") == ("T_box_max_peer_z", "T_box_max", "T_box")
    assert signal_aliases("TP2_rms_peer_delta") == ("TP2_rms_peer_delta", "TP2_rms", "TP2")
    assert signal_aliases("DV_pressure_std") == ("DV_pressure_std", "DV_pressure")
    assert signal_aliases("pos_err_rms") == ("pos_err_rms", "pos_err")
    # not a stat suffix, not a peer suffix: left alone
    assert signal_aliases("opening_time") == ("opening_time",)
    assert signal_aliases("i_rms_cruise") == ("i_rms_cruise",)
    assert signal_aliases("dwell_fraction_last_hour") == ("dwell_fraction_last_hour",)


def test_peer_normalised_signals_keep_the_sign_of_the_feature():
    """A positive peer delta means "hotter than its peers", i.e. the same direction as the
    feature itself - so it must classify the same way an unnormalised rise does."""
    plain = make_alert("bearing", [("vib_bpfo_mean", 5.0, 0.8)])
    peered = make_alert("bearing", [("vib_bpfo_mean_peer_z", 5.0, 0.8)])
    assert classify(plain)[0] == classify(peered)[0] == "outer_race"
    cold = make_alert("bearing", [("vib_bpfo_mean_peer_z", -5.0, 0.1)])
    assert classify(cold)[0] != "outer_race"


# --------------------------------------------------------------------------------------
# Evidence combination (fix round 1: friction used to misfire as backlash)
# --------------------------------------------------------------------------------------

#: The last scored row of the four real friction episodes in
#: ``data/scores/episodes.parquet`` (T01-door_L1-0, T05-door_R3-0, T08-door_L2-0,
#: T09-door_L3-0), signal names and z-scores verbatim.
REAL_FRICTION_EPISODES = {
    "T01-door_L1-0": [
        ("pos_err_rms", 182.5), ("pos_err_max", 136.4), ("pos_err_rms_peer_delta", 108.0),
        ("opening_time", 105.9), ("closing_time", 105.2),
    ],
    "T05-door_R3-0": [
        ("pos_err_rms", 142.7), ("pos_err_max", 109.9), ("pos_err_rms_peer_delta", 79.5),
        ("i_end", 77.1), ("opening_time", 75.9),
    ],
    "T08-door_L2-0": [
        ("pos_err_rms", 203.6), ("pos_err_max", 144.8), ("opening_time", 99.8),
        ("closing_time", 99.2), ("pos_err_rms_peer_delta", 98.9),
    ],
    "T09-door_L3-0": [
        ("pos_err_rms", 147.5), ("pos_err_max", 119.9), ("closing_time_peer_delta", -115.8),
        ("pos_err_max_peer_delta", -67.8), ("pos_err_rms_peer_delta", 66.3),
    ],
}


@pytest.mark.parametrize("episode", sorted(REAL_FRICTION_EPISODES))
def test_real_friction_episodes_classify_as_friction(episode):
    signals = [(name, z, 1.0) for name, z in REAL_FRICTION_EPISODES[episode]]
    alert = make_alert("door", signals, episode_id=episode, recent_events=[])
    assert classify(alert)[0] == "friction"


def test_backlash_requires_the_start_current_peak():
    """Position error alone is friction; add the start-current peak and it is backlash."""
    without = make_alert("door", [("pos_err_rms", 6.0, 0.03), ("pos_err_max", 5.0, 0.05)])
    assert classify(without)[0] == "friction"
    with_peak = make_alert(
        "door", [("i_start_peak", 6.0, 9.1), ("pos_err_rms", 5.0, 0.03)]
    )
    assert classify(with_peak)[0] == "backlash"


def test_evidence_combines_instead_of_strongest_signal_wins():
    """One big friction signal loses to three corroborating brush-wear signals."""
    alert = make_alert(
        "door",
        [("closing_time", 6.0, 3.9), ("i_peak", 5.5, 8.0), ("i_end", 5.0, 3.0),
         ("T_motor", 4.5, 70.0)],
    )
    assert classify(alert)[0] == "brush_wear"


def test_repeated_window_stats_of_one_channel_count_once():
    """TP2_rms / TP2_std / TP2_mean are one channel, not three votes."""
    one_channel = make_alert(
        "pneumatic",
        [("TP2_mean", 9.0, 10.1), ("TP2_rms", 8.5, 10.2), ("TP2_std", 8.0, 0.4),
         ("Flowmeter_max", 4.0, 60.0)],
    )
    # Flowmeter is the only other channel, so valve_leak_outlet must be able to compete
    assert classify(one_channel)[0] in {"valve_leak_outlet", "air_leak"}
    per_channel = make_alert(
        "pneumatic",
        [("Flowmeter_max", 9.0, 60.0), ("Flowmeter_mean", 8.5, 55.0), ("TP2_mean", 4.0, 10.1)],
    )
    assert classify(per_channel)[0] == "valve_leak_outlet"


def test_events_reach_the_faults_the_cycle_table_cannot_see():
    """The door cycle features carry no obstruction / limit-switch column - the event log
    does, so ``recent_events`` is the only path to those two faults."""
    blocked = make_alert("door", [], recent_events=["08:12 obstruction {\"pos\": 0.4}"])
    assert classify(blocked)[0] == "obstruction"
    switch = make_alert("door", [], recent_events=["08:12 ls_timeout"])
    assert classify(switch)[0] == "limit_switch"
    # a bare reversal is not enough on its own
    assert classify(make_alert("door", [], recent_events=["08:12 reversal"]))[0] == "unknown"


def test_bearing_never_gets_a_door_fault():
    """coerce_fault narrows to the subsystem, whatever the signals look like."""
    door_faults = set(FAULT_TYPES["door"]) - set(FAULT_TYPES["bearing"])
    for signals in (
        [("vib_rms_mean_peer_delta", -172.0, 0.01)],
        [("vib_crest_mean_peer_z", 2.3, 3.0), ("load_frac", 1.7, 0.4)],
        [("closing_time", 9.0, 3.9), ("i_start_peak", 8.0, 9.0)],
    ):
        fault, _ = classify(make_alert("bearing", signals))
        assert fault not in door_faults
        assert fault in set(FAULT_TYPES["bearing"]) | {"unknown"}


def test_bearing_thermal_split_by_rate():
    """dT_peer_same_side -> hot_axle_box when fast, bearing_degradation when slow."""
    fast = make_alert("bearing", [("dT_peer_same_side", 6.0, 14.0)], duration_h=2.0)
    slow = make_alert("bearing", [("dT_peer_same_side", 6.0, 14.0)], duration_h=72.0)
    assert classify(fast)[0] == "hot_axle_box"
    assert classify(slow)[0] == "bearing_degradation"
    # hot_axle_box is a safety stop, so it is always high urgency
    assert template_advisory(fast).urgency == "high"


def test_template_unknown_when_nothing_matches():
    alert = make_alert("door", [("some_unmodelled_column", 9.0, 1.0)])
    adv = template_advisory(alert)
    assert adv.likely_fault == "unknown"
    assert adv.confidence == pytest.approx(0.2)
    assert "watch" in adv.recommended_action.lower()


def test_template_no_signals_at_all():
    alert = make_alert("pneumatic", [], recent_events=[], fleet_peer_summary="")
    adv = template_advisory(alert)
    assert adv.likely_fault == "unknown"
    assert len(adv.evidence) >= 2


@pytest.mark.parametrize(
    "score, threshold, duration_h, expected",
    [
        (2.5, 2.4, 1.0, "low"),
        (3.2, 2.4, 1.0, "medium"),
        (2.5, 2.4, 8.0, "medium"),
        (5.9, 2.4, 6.2, "high"),
        (2.5, 2.4, 30.0, "high"),
    ],
)
def test_template_urgency(score, threshold, duration_h, expected):
    alert = make_alert(
        "door",
        [("closing_time", 3.0, 3.9)],
        score=score,
        threshold=threshold,
        duration_h=duration_h,
    )
    assert template_advisory(alert).urgency == expected


def test_template_fills_every_field_and_round_trips_json():
    for subsystem, signals in (
        ("door", [("closing_time", 5.1, 3.9)]),
        ("pneumatic", [("Reservoirs_mean", -5.0, 7.2)]),
        ("bearing", [("thermal_residual", 5.0, 9.0)]),
    ):
        adv = template_advisory(make_alert(subsystem, signals))
        assert adv.source == "template"
        assert adv.model == TEMPLATE_MODEL
        assert adv.summary and adv.likely_component and adv.recommended_action
        assert 2 <= len(adv.evidence) <= 5
        assert 0.0 <= adv.confidence <= 1.0
        assert set(adv.cbm_steps) == set(CBM_STEPS)
        assert all(adv.cbm_steps[k].strip() for k in CBM_STEPS)
        payload = json.loads(adv.model_dump_json())
        assert Advisory.model_validate(payload) == adv


def test_template_quotes_the_numbers_it_was_given():
    alert = make_alert("door", [("closing_time", 5.1, 3.9)])
    adv = template_advisory(alert)
    blob = " ".join(adv.evidence) + adv.summary + adv.cbm_steps["state_detection"]
    assert "5.9" in blob and "2.4" in blob and "closing_time" in blob


# --------------------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------------------


def test_coerce_fault_narrows_to_subsystem_and_never_raises():
    assert coerce_fault("friction", "door") == "friction"
    assert coerce_fault("air_leak", "door") == "unknown"  # wrong subsystem
    assert coerce_fault("Air Leak", "pneumatic") == "air_leak"
    assert coerce_fault("totally-made-up", "door") == "unknown"
    assert coerce_fault(None) == "unknown"
    assert coerce_fault("") == "unknown"


def test_advisory_validators_clip_and_coerce():
    adv = Advisory(
        summary="s",
        evidence=["a", "b", "c", "d", "e", "f", "  "],
        likely_component="c",
        likely_fault="not_a_fault",
        recommended_action="do it",
        urgency="URGENT",
        confidence=7.5,
        cbm_steps={"advisory": "go", "bogus": "drop me"},
    )
    assert len(adv.evidence) == 5
    assert adv.likely_fault == "unknown"
    assert adv.urgency == "medium"
    assert adv.confidence == 1.0
    assert set(adv.cbm_steps) == set(CBM_STEPS)
    assert Advisory(**{**adv.model_dump(), "confidence": -3}).confidence == 0.0


def test_alert_context_defaults_and_utc_coercion():
    alert = AlertContext(
        episode_id="e",
        train_id="T01",
        car=1,
        subsystem="bearing",
        component_id="axlebox_3R",
        model_name="m",
        score=1.0,
        threshold=0.5,
        t_start=datetime(2026, 9, 14, 8, 30),  # naive
    )
    assert alert.t_start.tzinfo is not None
    assert alert.t_start.utcoffset() == timedelta(0)
    assert alert.fault_candidates == [
        ft for ft in FAULT_TYPES["bearing"] if ft != "healthy"
    ]
    assert alert.glossary == GLOSSARY["bearing"]
    assert alert.ratio == pytest.approx(2.0)


def test_alert_context_zero_threshold_is_safe():
    alert = make_alert("door", [("closing_time", 3.0, 1.0)], threshold=0.0)
    assert alert.ratio == 1.0
    assert template_advisory(alert).urgency in ("low", "medium", "high")


# --------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------


def test_system_prompt_carries_the_cbm_steps_and_every_glossary():
    for step in CBM_STEPS:
        assert step in SYSTEM
    for subsystem, text in GLOSSARY.items():
        assert subsystem.upper() in SYSTEM
        assert text.splitlines()[0].strip("- ") in SYSTEM
    # grounded numbers that must survive any edit of the glossary
    assert "2 cm" in SYSTEM and "8.2 bar" in SYSTEM and "HABD" in SYSTEM


def test_user_message_contains_signals_numbers_candidates_and_glossary():
    alert = make_alert(
        "pneumatic",
        [("t_off", -5.0, 120.0), ("duty_ratio", 3.2, 0.41)],
    )
    msg = build_user_message(alert)
    assert "t_off" in msg and "-5.00" in msg and "120" in msg
    assert "duty_ratio" in msg and "+3.20" in msg
    assert "5.9" in msg and "2.4" in msg and "2.46x" in msg
    assert "2026-09-14T08:30:00Z" in msg
    assert "air_leak" in msg and "clogged_filter" in msg  # fault candidates
    assert "healthy" not in msg.split("FAULT CANDIDATES")[1]
    assert GLOSSARY["pneumatic"].splitlines()[0].strip("- ") in msg


def test_user_message_handles_empty_optional_blocks():
    alert = make_alert("door", [], recent_events=[], fleet_peer_summary="")
    msg = build_user_message(alert)
    assert "(none recorded)" in msg
    assert "(none in the window)" in msg
    assert "(no peer comparison supplied)" in msg


# --------------------------------------------------------------------------------------
# .env parsing / key discovery
# --------------------------------------------------------------------------------------


def test_parse_env_file(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "ANTHROPIC_API_KEY=sk-ant-plain  # trailing comment",
                'QUOTED="hello world"',
                "SINGLE='sq'",
                "export EXPORTED=yes",
                "  SPACED = spaced value ",
                "novalue",
                "=noname",
            ]
        ),
        encoding="utf-8",
    )
    env = parse_env_file(p)
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-plain"
    assert env["QUOTED"] == "hello world"
    assert env["SINGLE"] == "sq"
    assert env["EXPORTED"] == "yes"
    assert env["SPACED"] == "spaced value"
    assert "novalue" not in env and "" not in env


def test_parse_env_file_missing(tmp_path):
    assert parse_env_file(tmp_path / "nope") == {}


def test_find_api_key_prefers_process_env(tmp_path):
    p = tmp_path / ".env"
    p.write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
    assert find_api_key(env={"ANTHROPIC_API_KEY": "from-env"}, dotenv_path=p) == "from-env"
    assert find_api_key(env={}, dotenv_path=p) == "from-file"
    assert find_api_key(env={"ANTHROPIC_API_KEY": "   "}, dotenv_path=p) == "from-file"
    assert find_api_key(env={}, dotenv_path=tmp_path / "nope") is None


def test_repo_env_example_exists_and_has_no_secret():
    from nebulax.advisory.client import repo_root

    example = repo_root() / ".env.example"
    assert example.is_file()
    text = example.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY" in text
    assert "sk-ant-..." in text  # placeholder, not a key


# --------------------------------------------------------------------------------------
# advise(): client injection, failure and timeout
# --------------------------------------------------------------------------------------


DRAFT = AdvisoryDraft(
    summary="Door leaf L1 on T01 is dragging.",
    evidence=["closing_time z=+5.10", "score 5.9 vs threshold 2.4"],
    likely_component="door leaf door_L1 belt and guides",
    likely_fault="friction",
    recommended_action="Clean and re-lubricate the runners.",
    urgency="high",
    confidence=0.8,
    cbm_steps={
        "state_detection": "cusum scored 5.9 over 2.4",
        "health_assessment": "closing time and cruise current up together",
        "prognostic_assessment": "6.2 h open and climbing",
        "advisory": "Book the leaf for a guide inspection.",
    },
)


class _StubResponse:
    stop_reason = "end_turn"

    def __init__(self, parsed):
        self.parsed_output = parsed


class _FakeMessages:
    def __init__(self, behaviour, recorder):
        self._behaviour = behaviour
        self._recorder = recorder

    def parse(self, **kwargs):
        self._recorder.append(kwargs)
        return self._behaviour(kwargs)


class FakeClient:
    """Minimal stand-in for ``anthropic.Anthropic`` - no network, ever."""

    def __init__(self, behaviour):
        self.calls: list[dict] = []
        self.messages = _FakeMessages(behaviour, self.calls)


def test_advise_with_no_key_returns_template(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    alert = make_alert("door", [("closing_time", 5.1, 3.9)])
    adv = advise(alert, env={}, dotenv_path=tmp_path / "absent.env")
    assert adv.source == "template"
    assert adv.model == TEMPLATE_MODEL
    assert adv.likely_fault == "friction"


def test_advise_with_stub_client_returns_claude_advisory():
    client = FakeClient(lambda kw: _StubResponse(DRAFT))
    alert = make_alert("door", [("closing_time", 5.1, 3.9)])
    adv = advise(alert, client=client)
    assert adv.source == "claude"
    assert adv.model == CLAUDE_MODEL
    assert adv.likely_fault == "friction"
    assert adv.urgency == "high"
    assert set(adv.cbm_steps) == set(CBM_STEPS)
    assert adv.cbm_steps["advisory"] == "Book the leaf for a guide inspection."

    (kwargs,) = client.calls
    assert kwargs["model"] == CLAUDE_MODEL
    assert kwargs["max_tokens"] == 1024
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["output_format"] is AdvisoryDraft
    # adaptive thinking is on by default on Claude Opus 5 and would eat the frozen
    # 1024-token budget before the advisory is written
    assert kwargs["thinking"] == {"type": "disabled"}
    # the SDK gives up before advise()'s hard join timeout fires
    assert kwargs["timeout"] == pytest.approx(request_timeout(20.0))
    assert kwargs["timeout"] < 20.0
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["system"][0]["text"] == SYSTEM
    assert "closing_time" in kwargs["messages"][0]["content"]
    assert GLOSSARY["door"].splitlines()[0].strip("- ") in kwargs["messages"][0]["content"]


def test_sdk_timeout_stays_inside_the_join_timeout():
    client = FakeClient(lambda kw: _StubResponse(DRAFT))
    advise(make_alert("door", [("closing_time", 5.1, 3.9)]), client=client, timeout_s=5.0)
    (kwargs,) = client.calls
    assert kwargs["timeout"] == pytest.approx(4.0)
    # never below a second, however small the caller's budget
    assert request_timeout(0.25) == 1.0


def test_call_uses_with_options_to_disable_sdk_retries():
    """A retry would spend the rest of the budget; one attempt, then the template."""
    from nebulax.advisory import MAX_RETRIES, call_claude

    seen = {}

    class _Client(FakeClient):
        def with_options(self, **kw):
            seen.update(kw)
            return self

    client = _Client(lambda kw: _StubResponse(DRAFT))
    call_claude(make_alert("door", [("closing_time", 5.1, 3.9)]), client, timeout_s=20.0)
    assert MAX_RETRIES == 0
    assert seen["max_retries"] == 0
    assert seen["timeout"] == pytest.approx(request_timeout(20.0))


def test_narrowing_to_unknown_damps_the_confidence():
    """A fault the subsystem cannot have is not something to be 0.8 confident about."""
    bad = DRAFT.model_copy(update={"likely_fault": "air_leak", "confidence": 0.8})
    client = FakeClient(lambda kw: _StubResponse(bad))
    adv = advise(make_alert("door", [("closing_time", 5.1, 3.9)]), client=client)
    assert adv.likely_fault == "unknown"
    assert adv.confidence == pytest.approx(UNKNOWN_CONFIDENCE_CAP)
    # a legal fault keeps whatever confidence the model reported
    good = DRAFT.model_copy(update={"confidence": 0.8})
    ok = advise(
        make_alert("door", [("closing_time", 5.1, 3.9)]),
        client=FakeClient(lambda kw: _StubResponse(good)),
    )
    assert ok.likely_fault == "friction"
    assert ok.confidence == pytest.approx(0.8)


def test_narrow_to_leaves_a_low_confidence_unknown_alone():
    adv = Advisory(
        summary="s",
        likely_component="c",
        likely_fault="unknown",
        recommended_action="watch",
        confidence=0.2,
    )
    assert adv.narrow_to("door") is adv


def test_advise_narrows_a_wrong_subsystem_fault_to_unknown():
    bad = DRAFT.model_copy(update={"likely_fault": "air_leak"})  # pneumatic fault on a door
    client = FakeClient(lambda kw: _StubResponse(bad))
    adv = advise(make_alert("door", [("closing_time", 5.1, 3.9)]), client=client)
    assert adv.source == "claude"
    assert adv.likely_fault == "unknown"


def test_advise_falls_back_when_the_client_raises():
    client = FakeClient(lambda kw: (_ for _ in ()).throw(RuntimeError("boom")))
    adv = advise(make_alert("door", [("closing_time", 5.1, 3.9)]), client=client)
    assert adv.source == "template"


def test_advise_falls_back_when_there_is_no_parsed_output():
    client = FakeClient(lambda kw: _StubResponse(None))
    adv = advise(make_alert("door", [("closing_time", 5.1, 3.9)]), client=client)
    assert adv.source == "template"


def test_advise_honours_the_timeout_and_returns_the_template():
    started = threading.Event()

    def slow(_kw):
        started.set()
        time.sleep(30)  # never completes within the timeout; runs on a daemon thread
        raise AssertionError("should not be reached")

    client = FakeClient(slow)
    t0 = time.monotonic()
    adv = advise(make_alert("door", [("closing_time", 5.1, 3.9)]), client=client, timeout_s=0.25)
    elapsed = time.monotonic() - t0
    assert started.wait(5.0)
    assert adv.source == "template"
    assert elapsed < 5.0


def test_advise_never_raises_on_a_garbage_client():
    class Garbage:
        pass

    adv = advise(make_alert("bearing", [("thermal_residual", 5.0, 9.0)]), client=Garbage())
    assert adv.source == "template"
