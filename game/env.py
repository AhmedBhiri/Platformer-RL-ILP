from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import random


class Tile(str, Enum):
    GROUND = "ground"
    GAP = "gap"
    ENEMY = "enemy"


class Action(str, Enum):
    DO_NOTHING = "do_nothing"
    JUMP = "jump"
    ATTACK = "attack"


class Event(str, Enum):
    NONE = "none"
    CLEARED_ENEMY = "cleared_enemy"
    CLEARED_GAP = "cleared_gap"
    HIT_ENEMY = "hit_enemy"
    FELL_INTO_GAP = "fell_into_gap"
    WASTED_ATTACK = "wasted_attack"


@dataclass(frozen=True)
class Obs:
    """
    Observation is symbolic-friendly: discrete distances + player stance.
    Distances are ints in {1,2,3} or None meaning 'far / not in window'.
    """
    enemy_dist: Optional[int]   # 1..lookahead or None
    gap_dist: Optional[int]     # 1..lookahead or None
    on_ground: bool
    air_time: int               # remaining steps in air (0 => on ground)


@dataclass
class StepResult:
    obs: Obs
    reward: float
    done: bool
    event: Event
    info: Dict


class PlatformerEnv:
    """
    A tiny 1D platformer environment (no graphics).
    - World is a list of tiles that scrolls as the player moves forward.
    - Player can jump (air_time=2), attack (only works when enemy is 1 tile ahead),
      or do nothing.
    - If a gap or enemy reaches the player's tile while on ground => terminal.
    """

    def __init__(
        self,
        length: int = 200,
        lookahead: int = 3,
        jump_air_time: int = 2,
        p_gap: float = 0.08,
        p_enemy: float = 0.08,
        seed: Optional[int] = None,
    ) -> None:
        if lookahead < 1:
            raise ValueError("lookahead must be >= 1")
        self.length = length
        self.lookahead = lookahead
        self.jump_air_time = jump_air_time
        self.p_gap = p_gap
        self.p_enemy = p_enemy
        self.rng = random.Random(seed)

        # internal state
        self.track: List[Tile] = []
        self.t: int = 0                 # timestep
        self.player_x: int = 0          # index in track
        self.air_time: int = 0          # remaining air steps
        self.done: bool = False

        # for optional logging / analysis
        self.last_event: Event = Event.NONE

    def reset(self) -> Obs:
        self.track = self._generate_track(self.length)
        self.t = 0
        self.player_x = 0
        self.air_time = 0
        self.done = False
        self.last_event = Event.NONE
        return self._make_obs()

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode.")

        if not isinstance(action, Action):
            # allow passing strings
            try:
                action = Action(action)
            except Exception as e:
                raise ValueError(f"Invalid action: {action}") from e

        reward = 0.0
        event = Event.NONE

        # -------------------------------------------------
        # (0) Tick down airtime at START of step
        #     (so a new jump gets full duration)
        # -------------------------------------------------
        if self.air_time > 0:
            self.air_time -= 1

        # --- 1) Read current hazards in lookahead window ---
        enemy_dist = self._distance_to_next(Tile.ENEMY)
        gap_dist = self._distance_to_next(Tile.GAP)

        on_ground_before = (self.air_time == 0)
        # --- shaping: discourage staying on ground when a gap is imminent ---
        # If a gap is very close and you're on the ground, not jumping now is usually bad.

        # This makes "jump" become the best immediate-reward action in those states,
        # so Popper can learn pit rules.
        if gap_dist is not None and gap_dist <= 2 and on_ground_before and action != Action.JUMP:
            reward -= 2.0

        # discourage not attacking when an enemy is imminent (and you're on the ground)
        if enemy_dist is not None and enemy_dist <= 1 and on_ground_before and action != Action.ATTACK:
            reward -= 1.0

        # --- 2) Apply action effects (before scrolling) ---
        if action == Action.JUMP:
            # only jump if on ground
            if on_ground_before:
                self.air_time = self.jump_air_time

        elif action == Action.ATTACK:
            # only works if enemy is exactly 1 tile ahead NOW
            if enemy_dist == 1:
                # remove that enemy tile
                self.track[self.player_x + 1] = Tile.GROUND
                event = Event.CLEARED_ENEMY
                reward += 3.0
            else:
                event = Event.WASTED_ATTACK
                reward -= 0.5

        elif action == Action.DO_NOTHING:
            pass

        # --- 3) Advance world by one step (player moves forward) ---
        self.player_x += 1
        self.t += 1

        on_ground_after = (self.air_time == 0)

        # --- 4) Check what is on the player's tile now ---
        # If we reached end of track, end episode.
        if self.player_x >= len(self.track):
            self.done = True
            reward += 5.0
            return StepResult(
                obs=self._make_obs(),
                reward=reward,
                done=True,
                event=event,
                info={"t": self.t, "x": self.player_x,
                      "reason": "end_of_track"},
            )

        tile_now = self.track[self.player_x]

        # Terminal collisions only if on ground.
        if on_ground_after and tile_now == Tile.ENEMY:
            self.done = True
            event = Event.HIT_ENEMY
            reward -= 10.0

        elif on_ground_after and tile_now == Tile.GAP:
            self.done = True
            event = Event.FELL_INTO_GAP
            reward -= 10.0

        else:
            # Survived this step.
            reward += 1.0

            # Optional shaping: reward clearing hazards by being in air when crossing them.
            # (Only if not already set to cleared_enemy/wasted_attack)
            if event == Event.NONE:
                if tile_now == Tile.ENEMY and not on_ground_after:
                    event = Event.CLEARED_ENEMY
                    reward += 2.0
                elif tile_now == Tile.GAP and not on_ground_after:
                    event = Event.CLEARED_GAP
                    reward += 2.0

        self.last_event = event

        return StepResult(
            obs=self._make_obs(),
            reward=reward,
            done=self.done,
            event=event,
            info={
                "t": self.t,
                "x": self.player_x,
                "action": action.value,
                "enemy_dist_before": enemy_dist,
                "gap_dist_before": gap_dist,
                "on_ground_before": on_ground_before,
                "on_ground_after": on_ground_after,
                "air_time": self.air_time,
            },
        )

    # ---------- Helpers ----------

    def _generate_track(self, length: int) -> List[Tile]:
        """
        Generate a simple track:
        - Mostly ground.
        - Random single-tile gaps and enemies, spaced to avoid impossible situations.
        """
        track: List[Tile] = [Tile.GROUND] * length

        # Keep the first few tiles safe
        safe_prefix = max(5, self.lookahead + 2)

        i = safe_prefix
        while i < length - 1:
            r = self.rng.random()
            if r < self.p_gap:
                track[i] = Tile.GAP
                i += 2  # spacing
            elif r < self.p_gap + self.p_enemy:
                track[i] = Tile.ENEMY
                i += 2
            else:
                i += 1

        # Ensure start tile is safe
        track[0] = Tile.GROUND
        return track

    def _distance_to_next(self, tile_type: Tile) -> Optional[int]:
        """
        Distance from player to next tile of a given type within lookahead.
        Returns 1..lookahead, or None if not found.
        """
        for d in range(1, self.lookahead + 1):
            idx = self.player_x + d
            if idx < len(self.track) and self.track[idx] == tile_type:
                return d
        return None

    def _make_obs(self) -> Obs:
        return Obs(
            gap_dist=self._distance_to_next(Tile.GAP),
            enemy_dist=self._distance_to_next(Tile.ENEMY),
            on_ground=(self.air_time == 0),
            air_time=self.air_time,
        )
