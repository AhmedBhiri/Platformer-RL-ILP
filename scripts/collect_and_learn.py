from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow running as a module: `python -m scripts.collect_and_learn ...`
# while still being able to import `game.*`
if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from game.env import PlatformerEnv, Action  # noqa: E402


# -----------------------------
# Helpers
# -----------------------------
def obs_to_dict(obs: Any) -> Dict[str, Any]:
    if is_dataclass(obs):
        return asdict(obs)
    if hasattr(obs, "__dict__"):
        return dict(obs.__dict__)
    return {"obs": str(obs)}


def dist_value(x: Any) -> int:
    """
    Normalize distances into integers for BK.
      - int -> int
      - 'near' -> 1
      - 'far'  -> 99
      - None/unknown -> 99
    """
    if x is None:
        return 99
    if isinstance(x, bool):
        return 1 if x else 0
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s == "near":
            return 1
        if s == "far":
            return 99
        try:
            return int(s)
        except ValueError:
            return 99
    return 99


def action_to_atom(a: Action) -> str:
    if a == Action.DO_NOTHING:
        return "do_nothing"
    if a == Action.JUMP:
        return "jump"
    if a == Action.ATTACK:
        return "attack"
    return str(a).lower().replace(".", "_")


ALL_ACTIONS: List[Action] = [Action.DO_NOTHING, Action.JUMP, Action.ATTACK]


def oracle_good_action(enemy_d: int, gap_d: int, on_ground: bool) -> Action:
    """
    A simple, learnable teacher policy to label examples.
    Tune this later, but it gives Popper something consistent to learn.
    """
    # If a gap is imminent and we're grounded, jumping is "good"
    if on_ground and gap_d <= 1:
        return Action.JUMP
    # If an enemy is imminent, attacking is "good"
    if enemy_d <= 1:
        return Action.ATTACK
    # Otherwise do nothing
    return Action.DO_NOTHING


# -----------------------------
# Data collection
# -----------------------------
def collect_dataset(
    episodes: int,
    seed: int,
    max_steps_per_ep: int,
    eps_behavior: float,
) -> Tuple[List[Tuple[str, str, bool]], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      exs: list of (state_id, action_atom, is_positive)
      bk:  dict state_id -> features dict

    Guarantees:
      - state_ids are globally unique
      - no duplicate (state,action) lines
      - no contradictions (never both pos and neg for same (state,action))
    """
    rng = random.Random(seed)

    env = PlatformerEnv(seed=seed, length=300, lookahead=10,
                        p_gap=0.08, p_enemy=0.08)

    global_state_counter = 0

    # key: (state_id, action_atom) -> label
    seen_label: Dict[Tuple[str, str], bool] = {}

    exs: List[Tuple[str, str, bool]] = []
    bk: Dict[str, Dict[str, Any]] = {}

    for _ep in range(episodes):
        obs = env.reset()

        for _t in range(max_steps_per_ep):
            state_id = f"s{global_state_counter:09d}"
            global_state_counter += 1

            od = obs_to_dict(obs)
            enemy_d = dist_value(od.get("enemy_dist"))
            gap_d = dist_value(od.get("gap_dist"))
            on_ground = bool(od.get("on_ground", False))

            bk[state_id] = {
                "enemy_dist": enemy_d,
                "gap_dist": gap_d,
                "on_ground": on_ground,
            }

            # Compute oracle label for THIS state
            good = oracle_good_action(enemy_d, gap_d, on_ground)
            good_atom = action_to_atom(good)

            # --- Write ILP examples for this state ---
            # 1) Positive: the oracle action
            _emit_example(seen_label, exs, state_id, good_atom, True)

            # 2) Negatives: the other actions (optional but helps learning)
            for a in ALL_ACTIONS:
                a_atom = action_to_atom(a)
                if a_atom != good_atom:
                    _emit_example(seen_label, exs, state_id, a_atom, False)

            # --- Choose an action to actually execute in the env ---
            # eps-greedy around oracle so the rollouts aren't totally random
            if rng.random() < eps_behavior:
                act = rng.choice(ALL_ACTIONS)
            else:
                act = good

            step = env.step(act)
            obs = step.obs

            if step.done:
                break

    return exs, bk


def _emit_example(
    seen_label: Dict[Tuple[str, str], bool],
    exs: List[Tuple[str, str, bool]],
    state_id: str,
    act_atom: str,
    is_pos: bool,
) -> None:
    key = (state_id, act_atom)
    if key in seen_label:
        # if somehow called twice, do not allow contradictions
        return
    seen_label[key] = is_pos
    exs.append((state_id, act_atom, is_pos))


# -----------------------------
# Writing Popper task
# -----------------------------
def default_bias_text() -> str:
    # NOTE: unary types/directions must be (dist) and (in)
    return "\n".join(
        [
            "% Popper bias",
            "head_pred(good_action,2).",
            "body_pred(enemy_dist,2).",
            "body_pred(gap_dist,2).",
            "body_pred(on_ground,2).",
            "body_pred(near,1).",
            "body_pred(far,1).",
            "",
            "type(good_action,(state,action)).",
            "type(enemy_dist,(state,dist)).",
            "type(gap_dist,(state,dist)).",
            "type(on_ground,(state,bool)).",
            "type(near,(dist)).",
            "type(far,(dist)).",
            "",
            "direction(good_action,(in,in)).",
            "direction(enemy_dist,(in,out)).",
            "direction(gap_dist,(in,out)).",
            "direction(on_ground,(in,out)).",
            "direction(near,(in)).",
            "direction(far,(in)).",
            "",
            "action(do_nothing).",
            "action(jump).",
            "action(attack).",
            "",
            "max_body(4).",
            "max_vars(6).",
            "",
        ]
    ) + "\n"


def write_task(
    out_dir: Path,
    exs: List[Tuple[str, str, bool]],
    bk: Dict[str, Dict[str, Any]],
    bias_path: Path | None,
    overwrite_bias: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    bias_pl = out_dir / "bias.pl"
    bk_pl = out_dir / "bk.pl"
    exs_pl = out_dir / "exs.pl"

    # ---- bias.pl ----
    if bias_path is not None:
        # user-supplied bias file path
        if bias_path.exists():
            if overwrite_bias:
                bias_pl.write_text(bias_path.read_text())
                print(f"[bias] Overwrote {bias_pl} from {bias_path}.")
            else:
                # keep existing output bias if present; otherwise copy
                if bias_pl.exists():
                    print(
                        f"[bias] Keeping existing {bias_pl} (use --overwrite_bias to replace).")
                else:
                    bias_pl.write_text(bias_path.read_text())
                    print(f"[bias] Copied {bias_path} -> {bias_pl}.")
        else:
            # path provided but missing -> fall back to default
            bias_pl.write_text(default_bias_text())
            print(
                f"[bias] WARNING: {bias_path} not found; wrote default {bias_pl}.")
    else:
        # no bias path: write default (but do not overwrite unless asked)
        if bias_pl.exists() and not overwrite_bias:
            print(
                f"[bias] Keeping existing {bias_pl} (use --overwrite_bias to replace).")
        else:
            bias_pl.write_text(default_bias_text())
            print(f"[bias] Wrote default {bias_pl}.")

    # ---- bk.pl ----
    # Keep Prolog output quiet (Popper's recall parser is fragile w.r.t. warnings)
    lines = [
        ":- discontiguous enemy_dist/2.",
        ":- discontiguous gap_dist/2.",
        ":- discontiguous on_ground/2.",
        "",
        "% distance buckets",
        "near(D) :- integer(D), D =< 1.",
        "far(D)  :- integer(D), D >= 2.",
        "",
    ]
    for sid, feats in bk.items():
        lines.append(f"enemy_dist({sid},{feats['enemy_dist']}).")
        lines.append(f"gap_dist({sid},{feats['gap_dist']}).")
        lines.append(
            f"on_ground({sid},{'true' if feats['on_ground'] else 'false'}).")
    bk_pl.write_text("\n".join(lines) + "\n")

    # ---- exs.pl ----
    # Write in two blocks to avoid discontiguous pos/neg warnings
    pos_lines: List[str] = []
    neg_lines: List[str] = []
    for sid, act_atom, is_pos in exs:
        lit = f"good_action({sid},{act_atom})"
        if is_pos:
            pos_lines.append(f"pos({lit}).")
        else:
            neg_lines.append(f"neg({lit}).")

    exs_pl.write_text("\n".join(pos_lines + [""] + neg_lines) + "\n")


def run_popper(task_dir: Path, noisy: bool) -> int:
    cmd = [sys.executable, "Popper/popper.py", str(task_dir)]
    if noisy:
        cmd.append("--noisy")  # Popper supports this flag for noisy data
    print("Running:", " ".join(cmd))

    # Silence the pkg_resources deprecation warning (it can mess with parsers/logs)
    env = dict(os.environ)
    env.setdefault("PYTHONWARNINGS", "ignore")

    return subprocess.call(cmd, env=env)


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--out", type=str, default="ilp")

    ap.add_argument("--bias", type=str, default=None)
    ap.add_argument("--overwrite_bias", action="store_true")

    ap.add_argument("--run_popper", action="store_true")
    ap.add_argument("--noisy", action="store_true",
                    help="Run Popper with --noisy")

    ap.add_argument(
        "--eps_behavior",
        type=float,
        default=0.2,
        help="Epsilon for behavior policy during rollout (0=always oracle, 1=fully random).",
    )

    args = ap.parse_args()

    out_dir = Path(args.out)
    bias_path = Path(args.bias) if args.bias else None

    exs, bk = collect_dataset(
        episodes=args.episodes,
        seed=args.seed,
        max_steps_per_ep=args.max_steps,
        eps_behavior=args.eps_behavior,
    )

    write_task(out_dir, exs, bk, bias_path=bias_path,
               overwrite_bias=args.overwrite_bias)

    print(f"Wrote task to: {out_dir}")
    print(f"  bias.pl: {out_dir/'bias.pl'}")
    print(f"  bk.pl:   {out_dir/'bk.pl'}")
    print(f"  exs.pl:  {out_dir/'exs.pl'}")

    if args.run_popper:
        code = run_popper(out_dir, noisy=args.noisy)
        raise SystemExit(code)


if __name__ == "__main__":
    main()
