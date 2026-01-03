from __future__ import annotations

import argparse
import random
import subprocess
import sys
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow running as a script: `python scripts/collect_and_learn.py ...`
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
    Ensure distances become integers in BK.
    Your env might use ints or strings like 'far'. We normalize:
      - int -> int
      - 'far' -> 99
      - None -> 99
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
    # fallback
    return str(a).lower().replace(".", "_")


def choose_action_random(rng: random.Random) -> Action:
    return rng.choice([Action.DO_NOTHING, Action.JUMP, Action.ATTACK])


# -----------------------------
# Data collection
# -----------------------------
def collect_dataset(
    episodes: int,
    seed: int,
    max_steps_per_ep: int,
) -> Tuple[List[Tuple[str, str, bool]], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      exs: list of (state_id, action_atom, is_positive)
      bk:  dict state_id -> features dict
    """
    rng = random.Random(seed)

    env = PlatformerEnv(seed=seed, length=300, lookahead=10,
                        p_gap=0.08, p_enemy=0.08)

    # Global unique state id counter across ALL episodes
    global_state_counter = 0

    # To guarantee: never output same (state,action) twice, and never both pos+neg
    seen_label: Dict[Tuple[str, str], bool] = {}

    exs: List[Tuple[str, str, bool]] = []
    bk: Dict[str, Dict[str, Any]] = {}

    for ep in range(episodes):
        obs = env.reset()

        for _t in range(max_steps_per_ep):
            state_id = f"s{global_state_counter:09d}"
            global_state_counter += 1

            # Snapshot features for BK for THIS state_id
            od = obs_to_dict(obs)
            enemy_d = dist_value(od.get("enemy_dist"))
            gap_d = dist_value(od.get("gap_dist"))
            on_ground = bool(od.get("on_ground", False))

            bk[state_id] = {
                "enemy_dist": enemy_d,
                "gap_dist": gap_d,
                "on_ground": on_ground,
            }

            # Random behavior policy (for now)
            act = choose_action_random(rng)
            act_atom = action_to_atom(act)

            step = env.step(act)
            obs = step.obs

            # Label rule:
            # - treat strictly positive reward as "good_action"
            # - everything else as negative
            is_pos = bool(step.reward > 0)

            key = (state_id, act_atom)

            # This line makes contradictions impossible:
            # If we ever revisit the SAME (state,action) (shouldn't happen with unique state_id),
            # we force consistency (and do not duplicate lines).
            if key in seen_label:
                # If somehow it differs, keep the first label and skip the new one.
                if seen_label[key] != is_pos:
                    continue
                else:
                    continue

            seen_label[key] = is_pos
            exs.append((state_id, act_atom, is_pos))

            if step.done:
                break

    return exs, bk


# -----------------------------
# Writing Popper task
# -----------------------------
def write_task(out_dir: Path, exs: List[Tuple[str, str, bool]], bk: Dict[str, Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    bias_pl = out_dir / "bias.pl"
    bk_pl = out_dir / "bk.pl"
    exs_pl = out_dir / "exs.pl"

    # ---- bias.pl ----
    # NOTE: unary predicate types/directions MUST be (dist) and (in), not (dist,) / (in,)
    bias_pl.write_text(
        "\n".join(
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
        )
        + "\n"
    )

    # ---- bk.pl ----
    # Suppress discontiguous warnings completely (Popper's recall parser can get upset by warnings/noise).
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
    # Write in two blocks (all pos, then all neg) so SWI doesn't warn about discontiguous pos/neg.
    pos_lines = []
    neg_lines = []
    for sid, act_atom, is_pos in exs:
        lit = f"good_action({sid},{act_atom})"
        if is_pos:
            pos_lines.append(f"pos({lit}).")
        else:
            neg_lines.append(f"neg({lit}).")

    exs_pl.write_text("\n".join(pos_lines + [""] + neg_lines) + "\n")


def run_popper(task_dir: Path) -> int:
    cmd = [sys.executable, "Popper/popper.py", str(task_dir)]
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
    ap.add_argument("--run_popper", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)

    exs, bk = collect_dataset(
        episodes=args.episodes,
        seed=args.seed,
        max_steps_per_ep=args.max_steps,
    )

    write_task(out_dir, exs, bk)

    print(f"Wrote task to: {out_dir}")
    print(f"  bias.pl: {out_dir/'bias.pl'}")
    print(f"  bk.pl:   {out_dir/'bk.pl'}")
    print(f"  exs.pl:  {out_dir/'exs.pl'}")

    if args.run_popper:
        code = run_popper(out_dir)
        raise SystemExit(code)


if __name__ == "__main__":
    main()
