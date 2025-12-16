import argparse
import os
import random
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

from game.env import PlatformerEnv, Action, Event


@dataclass
class Example:
    state_id: str
    enemy_dist: str
    gap_dist: str
    on_ground: bool
    action: str
    label: str  # "pos" or "neg"


def dist_to_atom(d: Optional[int]) -> str:
    return str(d) if d is not None else "far"


def run_episode(env: PlatformerEnv, policy_eps: float, rng: random.Random) -> Tuple[int, List[Example]]:
    """
    Runs one episode, returns (distance_travelled, examples).
    policy_eps: epsilon for random exploration.
    """
    obs = env.reset()
    examples: List[Example] = []
    step_idx = 0

    while True:
        # epsilon-greedy: random action sometimes
        if rng.random() < policy_eps:
            action = rng.choice(
                [Action.DO_NOTHING, Action.JUMP, Action.ATTACK])
        else:
            # baseline "current policy" (initially simple, later replaced by ILP rules)
            action = simple_policy(obs)

        state_id = f"s{env.t:06d}_{step_idx:04d}"
        step_idx += 1

        step = env.step(action)

        # label only meaningful events
        label = None
        if step.event in (Event.CLEARED_GAP, Event.CLEARED_ENEMY):
            label = "pos"
        elif step.event in (Event.FELL_INTO_GAP, Event.HIT_ENEMY, Event.WASTED_ATTACK):
            label = "neg"

        if label:
            examples.append(
                Example(
                    state_id=state_id,
                    enemy_dist=dist_to_atom(
                        step.info.get("enemy_dist_before")),
                    gap_dist=dist_to_atom(step.info.get("gap_dist_before")),
                    on_ground=bool(step.info.get("on_ground_before")),
                    action=action.value,
                    label=label,
                )
            )

        obs = step.obs
        if step.done:
            return env.player_x, examples


def simple_policy(obs) -> Action:
    """
    Placeholder policy used when not exploring.
    Keep it VERY simple; this is not the learner.
    """
    # Attack if enemy is exactly 1 away
    if obs.enemy_dist == 1 and obs.on_ground:
        return Action.ATTACK
    # Jump if gap is exactly 1 away
    if obs.gap_dist == 1 and obs.on_ground:
        return Action.JUMP
    return Action.DO_NOTHING


def write_popper_files(examples: List[Example], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    bk_path = os.path.join(out_dir, "bk.pl")
    exs_path = os.path.join(out_dir, "exs.pl")
    bias_path = os.path.join(out_dir, "bias.pl")

    # --- Background knowledge ---
    # Keep it minimal for now: distances and boolean ground.
    with open(bk_path, "w", encoding="utf-8") as f:
        f.write("% Background knowledge\n")
        f.write("dist(1). dist(2). dist(3). dist(far).\n")
        f.write("near(1). near(2).\n")
        f.write("far(far).\n\n")

        # State facts
        for ex in examples:
            f.write(f"enemy_dist({ex.state_id},{ex.enemy_dist}).\n")
            f.write(f"gap_dist({ex.state_id},{ex.gap_dist}).\n")
            f.write(
                f"on_ground({ex.state_id},{'true' if ex.on_ground else 'false'}).\n")

    # --- Examples ---
    with open(exs_path, "w", encoding="utf-8") as f:
        for ex in examples:
            if ex.label == "pos":
                f.write(f"pos(good_action({ex.state_id},{ex.action})).\n")
            else:
                f.write(f"neg(good_action({ex.state_id},{ex.action})).\n")

    # --- Bias (Popper) ---
    # This is a conservative bias; we can tune later.
    with open(bias_path, "w", encoding="utf-8") as f:
        f.write("% Popper bias\n")
        f.write("head_pred(good_action,2).\n")
        f.write("body_pred(enemy_dist,2).\n")
        f.write("body_pred(gap_dist,2).\n")
        f.write("body_pred(on_ground,2).\n")
        f.write("body_pred(near,1).\n")
        f.write("body_pred(far,1).\n\n")

        f.write("type(good_action,(state,action)).\n")
        f.write("type(enemy_dist,(state,dist)).\n")
        f.write("type(gap_dist,(state,dist)).\n")
        f.write("type(on_ground,(state,bool)).\n")
        f.write("type(near,(dist,)).\n")
        f.write("type(far,(dist,)).\n\n")

        f.write("direction(good_action,(in,in)).\n")
        f.write("direction(enemy_dist,(in,out)).\n")
        f.write("direction(gap_dist,(in,out)).\n")
        f.write("direction(on_ground,(in,out)).\n")
        f.write("direction(near,(in,)).\n")
        f.write("direction(far,(in,)).\n\n")

        # allow these constants as actions
        f.write("action(do_nothing).\n")
        f.write("action(jump).\n")
        f.write("action(attack).\n\n")

        f.write("max_body(4).\n")
        f.write("max_vars(6).\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of episodes to run")
    parser.add_argument("--epsilon", type=float, default=0.5,
                        help="Exploration probability")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="ilp",
                        help="Output directory for Popper files")
    parser.add_argument("--run-popper", action="store_true",
                        help="Call Popper after generating files")
    parser.add_argument("--popper-path", type=str,
                        default="../Popper/popper.py", help="Path to popper.py")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = PlatformerEnv(seed=args.seed, length=300,
                        lookahead=3, p_gap=0.08, p_enemy=0.08)

    all_examples: List[Example] = []
    distances: List[int] = []

    for ep in range(args.episodes):
        dist, exs = run_episode(env, policy_eps=args.epsilon, rng=rng)
        distances.append(dist)
        all_examples.extend(exs)

    avg_dist = sum(distances) / max(1, len(distances))
    print(f"Ran {args.episodes} episodes. Avg distance = {avg_dist:.1f}. Collected examples = {len(all_examples)}")

    write_popper_files(all_examples, args.out)
    print(f"Wrote Popper files to: {args.out}/ (bk.pl, exs.pl, bias.pl)")

    if args.run_popper:
        # This assumes you cloned Popper somewhere; adjust popper path accordingly.
        cmd = ["python3", args.popper_path, args.out]
        print("Running Popper:", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
