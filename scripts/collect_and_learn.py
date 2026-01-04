from __future__ import annotations

import argparse
import copy
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow running as a script AND as module: `python -m scripts.collect_and_learn`
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
    Normalize distances to integers for BK facts.
    - int -> int
    - 'far' -> 99
    - 'near' -> 1
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
        if s == "far":
            return 99
        if s == "near":
            return 1
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


ACTIONS: List[Action] = [Action.DO_NOTHING, Action.JUMP, Action.ATTACK]


# -----------------------------
# Data collection
# -----------------------------
def collect_dataset(
    episodes: int,
    seed: int,
    max_steps_per_ep: int,
    label_best_action: bool = True,
) -> Tuple[List[Tuple[str, str, bool]], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      exs: list of (state_id, action_atom, is_positive)
      bk:  dict state_id -> features dict

    If label_best_action=True (recommended):
      - for each state, simulate all actions (via deepcopy(env))
      - pos = best reward action(s); neg = others
    Else:
      - take one random action and label by reward>0 (noisier)
    """
    rng = random.Random(seed)
    env = PlatformerEnv(seed=seed, length=300, lookahead=10,
                        p_gap=0.08, p_enemy=0.08)

    global_state_counter = 0

    # guarantee no duplicates, no contradictions
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

            if label_best_action:
                # Try to simulate all actions from the same state
                rewards: Dict[Action, float] = {}
                sim_ok = True
                for a in ACTIONS:
                    try:
                        env_sim = copy.deepcopy(env)
                        step_sim = env_sim.step(a)
                        rewards[a] = float(step_sim.reward)
                    except Exception:
                        sim_ok = False
                        break

                if sim_ok and rewards:
                    best = max(rewards.values())
                    # You can require best>0 if you want fewer positives. For now: best action(s) always positive.
                    for a in ACTIONS:
                        act_atom = action_to_atom(a)
                        is_pos = rewards[a] == best
                        key = (state_id, act_atom)
                        if key in seen_label:
                            continue
                        seen_label[key] = is_pos
                        exs.append((state_id, act_atom, is_pos))
                else:
                    # Fallback: one random action, label by reward>0
                    a = rng.choice(ACTIONS)
                    step = env.step(a)
                    obs = step.obs
                    act_atom = action_to_atom(a)
                    is_pos = bool(step.reward > 0)
                    key = (state_id, act_atom)
                    if key not in seen_label:
                        seen_label[key] = is_pos
                        exs.append((state_id, act_atom, is_pos))
                    if step.done:
                        break
            else:
                # Noisy mode: one random action per state
                a = rng.choice(ACTIONS)
                step = env.step(a)
                obs = step.obs
                act_atom = action_to_atom(a)
                is_pos = bool(step.reward > 0)
                key = (state_id, act_atom)
                if key not in seen_label:
                    seen_label[key] = is_pos
                    exs.append((state_id, act_atom, is_pos))
                if step.done:
                    break

            # advance the real env by taking a real action (so we keep moving)
            # If label_best_action=True we still need to actually step env once to progress.
            if label_best_action:
                a_real = rng.choice(ACTIONS)
                step_real = env.step(a_real)
                obs = step_real.obs
                if step_real.done:
                    break

    return exs, bk


# -----------------------------
# Writing Popper task
# -----------------------------
DEFAULT_BIAS = "\n".join(
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

    # bias.pl
    bias_pl = out_dir / "bias.pl"
    if bias_path is not None:
        # copy external bias into task dir
        text = Path(bias_path).read_text()
        if overwrite_bias or not bias_pl.exists():
            bias_pl.write_text(text)
        else:
            print(
                f"[bias] Keeping existing {bias_pl} (use --overwrite_bias to replace).")
    else:
        if overwrite_bias or not bias_pl.exists():
            bias_pl.write_text(DEFAULT_BIAS)
        else:
            print(
                f"[bias] Keeping existing {bias_pl} (use --overwrite_bias to replace).")

    # bk.pl
    bk_pl = out_dir / "bk.pl"
    lines = [
        ":- set_prolog_flag(verbose, silent).",
        ":- style_check(-discontiguous).",
        ":- style_check(-singleton).",
        ":- style_check(-var_branches).",
        "",
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

    # exs.pl
    exs_pl = out_dir / "exs.pl"
    pos_lines: List[str] = []
    neg_lines: List[str] = []

    for sid, act_atom, is_pos in exs:
        lit = f"good_action({sid},{act_atom})"
        (pos_lines if is_pos else neg_lines).append(
            f"{'pos' if is_pos else 'neg'}({lit}).")

    exs_pl.write_text(
        "\n".join(
            [
                ":- set_prolog_flag(verbose, silent).",
                ":- style_check(-discontiguous).",
                ":- style_check(-singleton).",
                "",
                "% positives",
                *pos_lines,
                "",
                "% negatives",
                *neg_lines,
                "",
            ]
        )
    )


def run_popper(task_dir: Path, popper_args: List[str]) -> int:
    cmd = [sys.executable, "Popper/popper.py", str(task_dir)] + popper_args
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--out", type=str, default="ilp")

    ap.add_argument("--bias", type=str, default=None,
                    help="Path to an existing bias.pl to use")
    ap.add_argument("--overwrite_bias", action="store_true",
                    help="Overwrite ilp/bias.pl if it exists")

    ap.add_argument(
        "--label_best_action",
        action="store_true",
        help="Label positives as the best immediate-reward action(s) per state (recommended).",
    )
    ap.add_argument(
        "--no_label_best_action",
        action="store_true",
        help="Disable best-action labeling and use reward>0 for one random action per state (noisier).",
    )

    ap.add_argument("--run_popper", action="store_true")
    ap.add_argument("--noisy", action="store_true",
                    help="Forward --noisy to Popper")

    args, unknown = ap.parse_known_args()

    label_best = True
    if args.no_label_best_action:
        label_best = False
    if args.label_best_action:
        label_best = True

    out_dir = Path(args.out)

    exs, bk = collect_dataset(
        episodes=args.episodes,
        seed=args.seed,
        max_steps_per_ep=args.max_steps,
        label_best_action=label_best,
    )

    bias_path = Path(args.bias) if args.bias else None
    write_task(out_dir, exs, bk, bias_path=bias_path,
               overwrite_bias=args.overwrite_bias)

    print(f"Wrote task to: {out_dir}")
    print(f"  bias.pl: {out_dir/'bias.pl'}")
    print(f"  bk.pl:   {out_dir/'bk.pl'}")
    print(f"  exs.pl:  {out_dir/'exs.pl'}")

    if args.run_popper:
        popper_args = []
        if args.noisy:
            popper_args.append("--noisy")
        popper_args += unknown
        code = run_popper(out_dir, popper_args)
        raise SystemExit(code)


if __name__ == "__main__":
    main()
