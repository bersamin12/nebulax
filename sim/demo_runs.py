"""First-cut physics simulators for the three subsystems, producing short healthy-vs-fault
runs for the 3D viewer.  Deliberately compact; the full simulators in nebulax/sim/ will
supersede these but keep the same signal names.

  python sim/demo_runs.py --out web/public/demo_runs.json
"""
import argparse
import json
import math

import numpy as np

# ----------------------------------------------------------------------------- door
def door_cycle(fault=None, dt=0.001, seed=0):
    """Bi-parting electric door: brushed DC motor -> gearbox -> belt -> leaf.  One leaf modelled;
    x in [0, 0.725] (0 = closed).  Returns per-sample dict of arrays + cycle features."""
    rng = np.random.default_rng(seed)
    R, kt, ke, G_r, eta, J = 4.0, 0.35, 0.35, 20 / 0.03, 0.75, 3e-5
    m_eff = 2 * 40 + J * G_r ** 2
    Fc0, b0, Vbus = 150.0, 250.0, 110.0
    X_OPEN, v_max, a_max = 0.725, 0.40, 0.8
    Fc, b, R_ = Fc0, b0, R
    x_obs = None
    if fault == "friction":
        Fc, b = Fc0 * 8.0, b0 * 3.0
    if fault == "brush":
        R_ = R * 1.8
    if fault == "obstruction":
        x_obs = 0.30
    if fault == "backlash":
        backlash = 0.008
    else:
        backlash = 0.0

    def trapezoid(t, x0, x1):
        """position/velocity reference from x0 to x1 with v_max/a_max."""
        d = abs(x1 - x0); sgn = 1 if x1 > x0 else -1
        ta = v_max / a_max
        if d < v_max * ta:            # triangular
            ta = math.sqrt(d / a_max); vm = a_max * ta; tc = 0
        else:
            vm = v_max; tc = (d - vm * ta) / vm
        T = 2 * ta + tc
        if t < ta:
            s, v = 0.5 * a_max * t * t, a_max * t
        elif t < ta + tc:
            s, v = 0.5 * a_max * ta * ta + vm * (t - ta), vm
        elif t < T:
            tr = T - t; s, v = d - 0.5 * a_max * tr * tr, a_max * tr
        else:
            s, v = d, 0.0
        return x0 + sgn * s, sgn * v, T

    t = 0.0; x = 0.0; v = 0.0; xm = 0.0
    integ = 0.0
    rows = {"t": [], "pos": [], "vel": [], "current": [], "pwm": [], "ls_open": [], "ls_closed": [], "obstruction": []}
    events = []
    phases = [("open", 0.0, X_OPEN), ("dwell", None, None), ("close", X_OPEN, 0.0)]
    retry_done = False
    for phase, x0, x1 in phases:
        if phase == "dwell":
            for _ in range(int(2.0 / dt)):
                t += dt
                _log(rows, t, x, 0.0, rng.normal(0, 0.03), 0.0, X_OPEN, 0)
            continue
        t_phase = 0.0
        while True:
            xr, vr, T = trapezoid(t_phase, x0, x1)
            # obstruction: obstacle at x_obs while closing, cleared after one retry
            F_obs = 0.0
            if x_obs is not None and phase == "close" and not retry_done and x < x_obs:
                F_obs = 20000.0 * (x_obs - x)
            # velocity PI + back-emf feedforward
            err = vr - v
            integ += err * dt
            V = 120.0 * err + 200.0 * integ + ke * G_r * vr + 2500.0 * (xr - x)
            V = max(-Vbus, min(Vbus, V))
            i = (V - ke * G_r * v) / R_
            F_motor = eta * kt * i * G_r
            F_fric = Fc * np.sign(v) + b * v if abs(v) > 1e-3 else np.clip(F_motor + F_obs, -Fc, Fc)
            a = (F_motor - F_fric + F_obs) / m_eff
            v += a * dt
            xm += v * dt
            # backlash: leaf follows motor-side position with dead band
            if backlash > 0:
                if xm - x > backlash / 2: x = xm - backlash / 2
                elif x - xm > backlash / 2: x = xm + backlash / 2
            else:
                x = xm
            x = max(-0.005, min(X_OPEN + 0.005, x))
            t += dt; t_phase += dt
            obstructed = 0
            if F_obs > 0 and (i > 6.0 or (xr - x) < -0.02):
                obstructed = 1
            _log(rows, t, x, v, i + rng.normal(0, 0.03), V / Vbus, X_OPEN, obstructed)
            if obstructed and not retry_done:
                events.append({"t": round(t, 2), "event": "obstruction detected, reversing"})
                # reverse 100 mm then retry
                for _ in range(int(1.2 / dt)):
                    t += dt
                    x = min(X_OPEN, x + 0.10 * dt / 1.2); xm = x
                    _log(rows, t, x, 0.083, 1.2 + rng.normal(0, 0.03), 0.3, X_OPEN, 1)
                for _ in range(int(1.0 / dt)):
                    t += dt
                    _log(rows, t, x, 0.0, 0.0, 0.0, X_OPEN, 0)
                retry_done = True
                x0, t_phase, integ = x, 0.0, 0.0
                continue
            done = (abs(x - x1) < 0.006 and abs(v) < 0.01 and t_phase > T)
            if done or t_phase > T + 4.0:
                if t_phase > T + 4.0:
                    events.append({"t": round(t, 2), "event": f"{phase} timed out"})
                break
    a = {k: np.array(vals) for k, vals in rows.items()}
    t_close_start = a["t"][np.argmax(a["vel"] < -0.01)]
    closed_idx = np.where((a["ls_closed"] == 1) & (a["t"] > t_close_start))[0]
    closing_time = float(a["t"][closed_idx[0]] - t_close_start) if len(closed_idx) else float("nan")
    feats = {
        "closing_time_s": round(closing_time, 2),
        "i_peak_A": round(float(a["current"].max()), 2),
        "i_mean_close_A": round(float(a["current"][a["vel"] < -0.05].mean()), 2) if (a["vel"] < -0.05).any() else None,
        "energy_J": round(float(np.sum(np.abs(a["current"] * a["pwm"] * 110.0)) * dt), 1),
        "obstruction_retries": int(sum(1 for e in events if "obstruction" in e["event"])),
    }
    step = 20  # 50 Hz out
    out = {k: [round(float(v_), 4) for v_ in a[k][::step]] for k in a}
    return {"signals": out, "features": feats, "events": events}

def _log(rows, t, x, v, i, pwm, X_OPEN, obstruction):
    rows["t"].append(t); rows["pos"].append(x); rows["vel"].append(v); rows["current"].append(i)
    rows["pwm"].append(pwm); rows["ls_open"].append(1 if x > X_OPEN - 0.008 else 0)
    rows["ls_closed"].append(1 if x < 0.008 else 0); rows["obstruction"].append(obstruction)

# ----------------------------------------------------------------------------- pneumatic
def pneumatic_run(fault=None, duration=1800, seed=0):
    rng = np.random.default_rng(seed)
    Q_comp, V_eff, P_atm = 9.0, 600.0, 1.013
    P_low, P_high, t_hold = 8.2, 10.0, 45.0
    Q_leak0, Q_aux = 0.3, 0.6
    leak_max = 3.0 if fault == "air_leak" else 0.0
    comp_scale = 0.6 if fault == "comp_wear" else 1.0
    hA, C_oil, T_amb = 18.0, 25000.0, 30.0
    P = 9.2; state = "OFF"; t_state = 0.0; T_oil = 62.0; tower = 0; t_tower = 0.0; purge = 0.0
    out = {k: [] for k in ("t", "TP2", "TP3", "Motor_current", "COMP", "Towers", "Oil_temperature", "LPS", "leak_rate", "brake")}
    events = []
    for t in range(duration):
        s = min(1.0, t / duration) ** 2 if leak_max else 0.0
        Q_leak = Q_leak0 + leak_max * s
        # service: 120 s cycle, braking phase 8 s from t%120 == 90
        braking = 1 if 90 <= (t % 120) < 98 else 0
        Q_brake = 120.0 / 8.0 if braking else 0.0
        Q_spring = 3.0 if (t % 120) < 10 else 0.0
        Q_in = Q_comp * comp_scale if state == "LOADED" else 0.0
        Q_purge = 0.15 * Q_comp if purge > 0 else 0.0
        P += P_atm * (Q_in - Q_leak - Q_brake - Q_spring - Q_aux - Q_purge) / V_eff
        # state machine
        t_state += 1
        if state in ("OFF", "UNLOADED") and P < P_low:
            state, t_state = "LOADED", 0; events.append({"t": t, "event": "compressor loaded"})
        elif state == "LOADED" and P > P_high:
            state, t_state = "UNLOADED", 0
        elif state == "UNLOADED" and t_state > t_hold:
            state, t_state = "OFF", 0
        if state == "LOADED":
            t_tower += 1
            if t_tower >= 90:
                tower ^= 1; t_tower = 0; purge = 10
        if purge > 0:
            purge -= 1
        if state == "OFF": I = 0.0
        elif state == "LOADED": I = (9.0 if t_state < 2 else 6.5 + 0.35 * (P - 8) + 0.02 * (T_oil - 60)) * (1.15 if fault == "comp_wear" else 1.0)
        else: I = 4.0
        P_heat = {"LOADED": 900.0, "UNLOADED": 350.0, "OFF": 0.0}[state] * (1.2 if fault == "comp_wear" else 1.0)
        T_oil += (P_heat - hA * (T_oil - T_amb)) / C_oil
        out["t"].append(t); out["TP2"].append(round(P + 0.3 + rng.normal(0, 0.01), 3) if state == "LOADED" else round(rng.normal(0.02, 0.01), 3))
        out["TP3"].append(round(P + rng.normal(0, 0.01), 3)); out["Motor_current"].append(round(I + rng.normal(0, 0.05), 2))
        out["COMP"].append(0 if state == "LOADED" else 1); out["Towers"].append(tower)
        out["Oil_temperature"].append(round(T_oil + rng.normal(0, 0.1), 2)); out["LPS"].append(1 if P < 7.0 else 0)
        out["leak_rate"].append(round(Q_leak, 2)); out["brake"].append(braking)
    comp = np.array(out["COMP"])
    duty = float(1 - comp.mean())
    feats = {"duty_cycle": round(duty, 3), "loaded_runs": int(np.sum(np.diff(comp) == -1)),
             "min_TP3_bar": round(float(min(out["TP3"])), 2), "max_oil_C": round(float(max(out["Oil_temperature"])), 1)}
    return {"signals": out, "features": feats, "events": events[:6]}

# ----------------------------------------------------------------------------- bearing
def bearing_run(fault=None, duration=3600, seed=0):
    rng = np.random.default_rng(seed)
    r_wheel, d_bore, mu0, k_visc = 0.425, 0.13, 0.0018, 0.15
    C_box, hA0, hA1 = 19000.0, 8.0, 0.4
    boxes = [f"{a}{s}" for a in (1, 2, 3, 4) for s in "LR"]
    scatter = rng.normal(1.0, 0.05, size=8)
    T = np.full(8, 34.0)
    T_amb = 30.0
    load_frac = 0.6
    speed = 0.0
    out = {"t": [], "speed": [], "T_amb": []}
    for bx in boxes:
        out[f"T_{bx}"] = []; out[f"vib_rms_{bx}"] = []; out[f"vib_kurt_{bx}"] = []
    bad = boxes.index("3R") if fault == "bearing" else None
    for t in range(duration):
        # station cycle: 120 s run (accel 22 s, cruise, brake 22 s) + 30 s dwell
        ph = t % 150
        if ph < 22: speed = min(22.0, ph * 1.0)
        elif ph < 98: speed = 22.0
        elif ph < 120: speed = max(0.0, 22.0 - (ph - 98) * 1.0)
        else: speed = 0.0
        s_deg = min(1.0, t / duration) ** 2 if bad is not None else 0.0
        omega = speed / r_wheel
        F_box = (10 + 6 * load_frac) / 2 * 9810.0
        for k in range(8):
            mu = mu0 * scatter[k] * (1 + 4 * s_deg if k == bad else 1.0)
            M = 0.5 * mu * F_box * d_bore + k_visc * max(omega, 1.0) ** (2 / 3)
            P = M * omega
            hA = (hA0 + hA1 * speed) * (0.9 + 0.2 * rng.random() * 0.1 + 0.05 * (k % 3))
            T[k] += (P - hA * (T[k] - T_amb)) / C_box
        if t % 5 == 0:
            out["t"].append(t); out["speed"].append(round(speed, 1)); out["T_amb"].append(T_amb)
            for k, bx in enumerate(boxes):
                sd = s_deg if k == bad else 0.0
                out[f"T_{bx}"].append(round(float(T[k] + rng.normal(0, 0.3)), 2))
                out[f"vib_rms_{bx}"].append(round(float(1.2 * (speed / 22) ** 1.2 * (1 + 3 * sd) * (1 + rng.normal(0, 0.03))), 3))
                out[f"vib_kurt_{bx}"].append(round(float(3 + 6 * sd * sd + rng.normal(0, 0.2)), 2))
    Tm = {bx: np.array(out[f"T_{bx}"]) for bx in boxes}
    peers = np.median(np.stack([Tm[bx] for bx in boxes]), axis=0)
    feats = {"max_T_C": round(float(max(v.max() for v in Tm.values())), 1),
             "max_dT_peer_K": round(float(max((Tm[bx] - peers).max() for bx in boxes)), 1)}
    return {"signals": out, "features": feats, "events": []}

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/public/demo_runs.json")
    args = ap.parse_args()
    data = {
        "door": {"healthy": door_cycle(None), "friction": door_cycle("friction"), "obstruction": door_cycle("obstruction")},
        "pneumatic": {"healthy": pneumatic_run(None), "air_leak": pneumatic_run("air_leak")},
        "bearing": {"healthy": bearing_run(None), "bearing": bearing_run("bearing")},
    }
    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    for sub, runs in data.items():
        for name, run in runs.items():
            print(sub, name, run["features"], run["events"][:2])
    import os
    print("wrote", args.out, round(os.path.getsize(args.out) / 1e3), "KB")

if __name__ == "__main__":
    main()
