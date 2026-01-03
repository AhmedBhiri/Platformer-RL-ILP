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


def run_episode(env: PlatformerEnv, policy_eps: float, rng: random.Random, episode_id: int) -> Tuple[int, List[Example]]:
    """
    Runs one episode, returns (distance_travelled, examples).
    policy_eps: epsilon for random exploration.
    episode_id: unique episode identifier to ensure unique state IDs.
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

        # Include episode_id to make state IDs globally unique
        state_id = f"s{episode_id:04d}_{step_idx:04d}"
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

    # Deduplicate examples: for BK facts, we only need one per state_id
    # For examples, we need to handle conflicts (same state+action with different labels)
    seen_states = {}  # state_id -> (enemy_dist, gap_dist, on_ground)
    example_labels = {}  # (state_id, action) -> {"pos": count, "neg": count}

    for ex in examples:
        # Track state facts (should be consistent for the same state_id now)
        if ex.state_id not in seen_states:
            seen_states[ex.state_id] = (ex.enemy_dist, ex.gap_dist, ex.on_ground)
        
        # Track example labels
        key = (ex.state_id, ex.action)
        if key not in example_labels:
            example_labels[key] = {"pos": 0, "neg": 0}
        example_labels[key][ex.label] += 1

    # --- Background knowledge ---
    with open(bk_path, "w", encoding="utf-8") as f:
        f.write("% Background knowledge\n")
        f.write("dist(1). dist(2). dist(3). dist(far).\n")
        f.write("near(1). near(2).\n")
        f.write("far(far).\n\n")

        # State facts (deduplicated)
        for state_id, (enemy_dist, gap_dist, on_ground) in sorted(seen_states.items()):
            f.write(f"enemy_dist({state_id},{enemy_dist}).\n")
            f.write(f"gap_dist({state_id},{gap_dist}).\n")
            f.write(f"on_ground({state_id},{'true' if on_ground else 'false'}).\n")
            # Add derived predicates for specific conditions
            if enemy_dist == "1":
                f.write(f"enemy_near({state_id}).\n")
            if gap_dist == "1":
                f.write(f"gap_near({state_id}).\n")
            if on_ground:
                f.write(f"grounded({state_id}).\n")

    # --- Examples ---
    # Resolve conflicts: if both pos and neg exist, skip that example (ambiguous)
    # Create separate predicates for each action: should_attack, should_jump, should_do_nothing
    with open(exs_path, "w", encoding="utf-8") as f:
        skipped = 0
        for (state_id, action), counts in sorted(example_labels.items()):
            if counts["pos"] > 0 and counts["neg"] > 0:
                # Conflict - skip this example
                skipped += 1
                continue
            # Map action to predicate name
            action_pred = f"should_{action}"  # e.g., should_attack, should_jump
            if counts["pos"] > 0:
                f.write(f"pos({action_pred}({state_id})).\n")
            else:
                f.write(f"neg({action_pred}({state_id})).\n")
        if skipped > 0:
            print(f"Warning: Skipped {skipped} conflicting examples")

    # --- Bias (Popper) ---
    # Use separate head predicates for each action
    with open(bias_path, "w", encoding="utf-8") as f:
        f.write("% Popper bias\n")
        f.write("max_body(3).\n")
        f.write("max_vars(4).\n\n")
        
        f.write("head_pred(should_attack,1).\n")
        f.write("head_pred(should_jump,1).\n")
        f.write("body_pred(enemy_near,1).\n")
        f.write("body_pred(gap_near,1).\n")
        f.write("body_pred(grounded,1).\n")
        f.write("body_pred(enemy_dist,2).\n")
        f.write("body_pred(gap_dist,2).\n")
        f.write("body_pred(on_ground,2).\n")


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
        dist, exs = run_episode(env, policy_eps=args.epsilon, rng=rng, episode_id=ep)
        distances.append(dist)
        all_examples.extend(exs)

    avg_dist = sum(distances) / max(1, len(distances))
    print(f"Ran {args.episodes} episodes. Avg distance = {avg_dist:.1f}. Collected examples = {len(all_examples)}")

    write_popper_files(all_examples, args.out)
    print(f"Wrote Popper files to: {args.out}/ (bk.pl, exs.pl, bias.pl)")

    if args.run_popper:
        # This assumes you cloned Popper somewhere; adjust popper path accordingly.
        import sys
        cmd = [sys.executable, args.popper_path, args.out]
        print("Running Popper:", " ".join(cmd))
        # Note: Popper uses SIGALRM which is Unix-only.
        # On Windows, you may need to run Popper in WSL or patch loop.py
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
