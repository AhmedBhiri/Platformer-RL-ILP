from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import random


class Tile(str, Enum):
    GROUND = "ground"
    GAP = "gap"
    ENEMY = "enemy"


class Action(str, Enum):
    DO_NOTHING = "do_nothing"
    JUMP = "jump"
    ATTACK = "attack"
    DUCK = "duck"


class ProjectileHeight(str, Enum):
    LOW = "low"
    HIGH = "high"


class Event(str, Enum):
    NONE = "none"
    CLEARED_ENEMY = "cleared_enemy"
    CLEARED_GAP = "cleared_gap"
    HIT_ENEMY = "hit_enemy"
    FELL_INTO_GAP = "fell_into_gap"
    WASTED_ATTACK = "wasted_attack"
    HIT_PROJECTILE = "hit_projectile"
    DODGED_PROJECTILE = "dodged_projectile"


@dataclass(frozen=True)
class Obs:
    """
    Distances are ints in {1..lookahead} or None meaning 'not in window'.
    """
    enemy_dist: Optional[int]
    gap_dist: Optional[int]
    projectile_dist: Optional[int]
    projectile_low: bool
    projectile_high: bool
    on_ground: bool
    air_time: int


@dataclass
class StepResult:
    obs: Obs
    reward: float
    done: bool
    event: Event
    info: Dict


class PlatformerEnv:
    """
    1D platformer + projectiles.

    Projectile rules:
      - LOW projectile: must JUMP (be in air) to dodge
      - HIGH projectile: must DUCK to dodge

    Enemy rules:
      - ATTACK clears enemy at dist=1 (before scrolling)
      - enemy on your tile while on ground => death

    Gap rules:
      - gap on your tile while on ground => death
      - jumping lets you clear gaps

    DUCK:
      - one-step stance (ducking_this_step)
    """

    def __init__(
        self,
        length: int = 200,
        lookahead: int = 3,
        jump_air_time: int = 2,
        p_gap: float = 0.08,
        p_enemy: float = 0.08,
        p_shoot: float = 0.15,     # enemy shoots chance when visible
        p_low_proj: float = 0.5,   # LOW vs HIGH projectile mix
        seed: Optional[int] = None,
    ) -> None:
        if lookahead < 1:
            raise ValueError("lookahead must be >= 1")

        self.length = length
        self.lookahead = lookahead
        self.jump_air_time = jump_air_time
        self.p_gap = p_gap
        self.p_enemy = p_enemy
        self.p_shoot = p_shoot
        self.p_low_proj = p_low_proj
        self.rng = random.Random(seed)

        self.track: List[Tile] = []
        self.t: int = 0
        self.player_x: int = 0
        self.air_time: int = 0
        self.done: bool = False

        # projectiles: list of (x, height)
        self.projectiles: List[Tuple[int, ProjectileHeight]] = []

        self.last_event: Event = Event.NONE

    def reset(self) -> Obs:
        self.track = self._generate_track(self.length)
        self.t = 0
        self.player_x = 0
        self.air_time = 0
        self.done = False
        self.projectiles = []
        self.last_event = Event.NONE
        return self._make_obs()

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode.")

        if not isinstance(action, Action):
            try:
                action = Action(action)
            except Exception as e:
                raise ValueError(f"Invalid action: {action}") from e

        reward = 0.0
        event = Event.NONE

        ducking_this_step = (action == Action.DUCK)

        # (0) Tick down airtime at START (so new jump gets full duration)
        if self.air_time > 0:
            self.air_time -= 1

        # (1) Read hazards
        enemy_dist = self._distance_to_next(Tile.ENEMY)
        gap_dist = self._distance_to_next(Tile.GAP)
        proj_dist, proj_h = self._nearest_projectile_in_view()

        on_ground_before = (self.air_time == 0)

        # (1.5) Enemies may shoot (spawn projectile at enemy position)
        if enemy_dist is not None:
            enemy_x = self.player_x + enemy_dist
            if self.rng.random() < self.p_shoot:
                h = ProjectileHeight.LOW if self.rng.random(
                ) < self.p_low_proj else ProjectileHeight.HIGH
                if (enemy_x, h) not in self.projectiles:
                    self.projectiles.append((enemy_x, h))

        # (1.6) Reward shaping (keep yours, add projectile-specific preference)
        if gap_dist is not None and gap_dist <= 2 and on_ground_before and action != Action.JUMP:
            reward -= 2.0
        if enemy_dist is not None and enemy_dist <= 1 and action != Action.ATTACK:
            reward -= 3.0

        # Projectile shaping: prefer the correct response depending on height
        if proj_dist is not None and proj_dist <= 1:
            if proj_h == ProjectileHeight.LOW:
                if action == Action.JUMP:
                    reward += 1.0
                elif action != Action.JUMP:
                    reward -= 1.5
            elif proj_h == ProjectileHeight.HIGH:
                if action == Action.DUCK:
                    reward += 1.0
                elif action != Action.DUCK:
                    reward -= 1.5

        # (2) Apply action effects (before scrolling)
        if action == Action.JUMP:
            if on_ground_before:
                self.air_time = self.jump_air_time

        elif action == Action.ATTACK:
            if enemy_dist == 1 and (self.player_x + 1) < len(self.track):
                self.track[self.player_x + 1] = Tile.GROUND
                event = Event.CLEARED_ENEMY
                reward += 3.0
            else:
                event = Event.WASTED_ATTACK
                reward -= 0.5

        elif action == Action.DUCK:
            reward -= 0.05  # tiny cost to stop random duck spam

        # (3) Advance world
        self.player_x += 1
        self.t += 1

        # move projectiles toward player by 1 each step
        self.projectiles = [(px - 1, h) for (px, h) in self.projectiles]

        on_ground_after = (self.air_time == 0)
        in_air = not on_ground_after

        # (4) End of track
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

        # (5) Projectile collision first (can be dodged)
        hit_h = None
        for (px, h) in self.projectiles:
            if px == self.player_x:
                hit_h = h
                break

        if hit_h is not None:
            ok = False
            if hit_h == ProjectileHeight.LOW:
                ok = in_air
            elif hit_h == ProjectileHeight.HIGH:
                ok = ducking_this_step

            if ok:
                self.projectiles = [(px, h) for (
                    px, h) in self.projectiles if px != self.player_x]
                if event == Event.NONE:
                    event = Event.DODGED_PROJECTILE
                reward += 1.0
            else:
                self.done = True
                event = Event.HIT_PROJECTILE
                reward -= 10.0

        # (6) Normal hazards
        if not self.done:
            if on_ground_after and tile_now == Tile.ENEMY:
                self.done = True
                event = Event.HIT_ENEMY
                reward -= 10.0
            elif on_ground_after and tile_now == Tile.GAP:
                self.done = True
                event = Event.FELL_INTO_GAP
                reward -= 10.0
            else:
                reward += 1.0
                if event == Event.NONE:
                    if tile_now == Tile.ENEMY and in_air:
                        event = Event.CLEARED_ENEMY
                        reward += 2.0
                    elif tile_now == Tile.GAP and in_air:
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
                "projectile_dist_before": proj_dist,
                "projectile_height_before": (proj_h.value if proj_h else None),
                "ducking": ducking_this_step,
                "on_ground_before": on_ground_before,
                "on_ground_after": on_ground_after,
                "air_time": self.air_time,
                "num_projectiles": len(self.projectiles),
            },
        )

    # ---------- Helpers ----------

    def _generate_track(self, length: int) -> List[Tile]:
        track: List[Tile] = [Tile.GROUND] * length
        safe_prefix = max(5, self.lookahead + 2)

        i = safe_prefix
        while i < length - 1:
            r = self.rng.random()
            if r < self.p_gap:
                track[i] = Tile.GAP
                i += 2
            elif r < self.p_gap + self.p_enemy:
                track[i] = Tile.ENEMY
                i += 2
            else:
                i += 1

        track[0] = Tile.GROUND
        return track

    def _distance_to_next(self, tile_type: Tile) -> Optional[int]:
        for d in range(1, self.lookahead + 1):
            idx = self.player_x + d
            if idx < len(self.track) and self.track[idx] == tile_type:
                return d
        return None

    def _nearest_projectile_in_view(self) -> Tuple[Optional[int], Optional[ProjectileHeight]]:
        for d in range(1, self.lookahead + 1):
            x = self.player_x + d
            for (px, h) in self.projectiles:
                if px == x:
                    return d, h
        return None, None

    def _make_obs(self) -> Obs:
        proj_d, proj_h = self._nearest_projectile_in_view()
        return Obs(
            gap_dist=self._distance_to_next(Tile.GAP),
            enemy_dist=self._distance_to_next(Tile.ENEMY),
            projectile_dist=proj_d,
            projectile_low=(proj_h == ProjectileHeight.LOW),
            projectile_high=(proj_h == ProjectileHeight.HIGH),
            on_ground=(self.air_time == 0),
            air_time=self.air_time,
        )
