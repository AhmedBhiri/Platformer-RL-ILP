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
    CLEARED_PROJECTILE = "cleared_projectile"
    HIT_PROJECTILE = "hit_projectile"


@dataclass(frozen=True)
class Obs:
    """
    Observation is symbolic-friendly: discrete distances + player stance.
    Distances are ints in {1,2,3} or None meaning 'far / not in window'.
    """
    enemy_dist: Optional[int]        # 1..lookahead or None
    gap_dist: Optional[int]          # 1..lookahead or None
    projectile_dist: Optional[int]   # 1..lookahead or None
    on_ground: bool
    air_time: int                    # remaining steps in air (0 => on ground)


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
    - Player can jump, attack, or do nothing.
    - Enemies can fire projectiles that travel toward the player.
    - ATTACK can clear:
        * enemy at distance 1
        * projectile at distance 1
    - If a gap, enemy, or projectile reaches the player's tile while on ground => terminal.
    """

    def __init__(
        self,
        length: int = 200,
        lookahead: int = 3,
        jump_air_time: int = 2,
        p_gap: float = 0.08,
        p_enemy: float = 0.08,
        p_shoot: float = 0.15,  # chance an enemy shoots when in view
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
        self.rng = random.Random(seed)

        # internal state
        self.track: List[Tile] = []
        self.t: int = 0                 # timestep
        self.player_x: int = 0          # index in track
        self.air_time: int = 0          # remaining air steps
        self.done: bool = False

        # enemy projectiles (absolute x positions in track coordinates)
        self.projectiles: List[int] = []

        # for optional logging / analysis
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
        projectile_dist = self._distance_to_next_projectile()

        on_ground_before = (self.air_time == 0)

        # --- (1.5) Enemies may shoot (spawn projectile at enemy position) ---
        # Keep it simple: if an enemy is visible, it may shoot a projectile.
        if enemy_dist is not None:
            enemy_x = self.player_x + enemy_dist
            if self.rng.random() < self.p_shoot:
                # avoid spawning duplicates at same position
                if enemy_x not in self.projectiles:
                    self.projectiles.append(enemy_x)

        # --- shaping: discourage staying on ground when a gap is imminent ---
        if gap_dist is not None and gap_dist <= 2 and on_ground_before and action != Action.JUMP:
            reward -= 2.0

        # discourage not attacking when an enemy is imminent
        if enemy_dist is not None and enemy_dist <= 1 and action != Action.ATTACK:
            reward -= 3.0

        # discourage not attacking when a projectile is imminent
        if projectile_dist is not None and projectile_dist <= 1 and action != Action.ATTACK:
            reward -= 2.0

        # --- 2) Apply action effects (before scrolling) ---
        if action == Action.JUMP:
            # only jump if on ground
            if on_ground_before:
                self.air_time = self.jump_air_time

        elif action == Action.ATTACK:
            cleared_something = False

            # clear enemy if exactly 1 tile ahead NOW
            if enemy_dist == 1 and (self.player_x + 1) < len(self.track):
                self.track[self.player_x + 1] = Tile.GROUND
                reward += 3.0
                event = Event.CLEARED_ENEMY
                cleared_something = True

            # also clear projectile if exactly 1 tile ahead NOW
            if (self.player_x + 1) in self.projectiles:
                self.projectiles = [
                    p for p in self.projectiles if p != (self.player_x + 1)]
                reward += 2.0
                # if we didn't already set event to cleared_enemy, use cleared_projectile
                if event == Event.NONE:
                    event = Event.CLEARED_PROJECTILE
                cleared_something = True

            if not cleared_something:
                event = Event.WASTED_ATTACK
                reward -= 0.5

        elif action == Action.DO_NOTHING:
            pass

        # --- 3) Advance world by one step (player moves forward) ---
        self.player_x += 1
        self.t += 1

        # projectiles travel toward the player by 1 each step
        # (absolute positions shift left in world coordinates relative to the player)
        self.projectiles = [p - 1 for p in self.projectiles]

        on_ground_after = (self.air_time == 0)

        # --- 4) Check end of track ---
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

        # --- 5) Check what is on the player's tile now ---
        tile_now = self.track[self.player_x]

        # Projectile collision (terminal only if on ground, like other hazards)
        if on_ground_after and self.player_x in self.projectiles:
            self.done = True
            event = Event.HIT_PROJECTILE
            reward -= 10.0

        # Terminal collisions only if on ground.
        elif on_ground_after and tile_now == Tile.ENEMY:
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
            # (Only if not already set to cleared_enemy/wasted_attack/cleared_projectile)
            if event == Event.NONE:
                if tile_now == Tile.ENEMY and not on_ground_after:
                    event = Event.CLEARED_ENEMY
                    reward += 2.0
                elif tile_now == Tile.GAP and not on_ground_after:
                    event = Event.CLEARED_GAP
                    reward += 2.0
                # if projectile reaches your tile while you're in air, treat as "cleared"
                elif self.player_x in self.projectiles and not on_ground_after:
                    # remove it (you "dodged" it)
                    self.projectiles = [
                        p for p in self.projectiles if p != self.player_x]
                    event = Event.CLEARED_PROJECTILE
                    reward += 1.0

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
                "projectile_dist_before": projectile_dist,
                "on_ground_before": on_ground_before,
                "on_ground_after": on_ground_after,
                "air_time": self.air_time,
                "num_projectiles": len(self.projectiles),
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

    def _distance_to_next_projectile(self) -> Optional[int]:
        """
        Distance from player to next projectile within lookahead.
        Returns 1..lookahead, or None if not found.
        """
        if not self.projectiles:
            return None
        # projectiles are absolute positions; find nearest ahead
        for d in range(1, self.lookahead + 1):
            if (self.player_x + d) in self.projectiles:
                return d
        return None

    def _make_obs(self) -> Obs:
        return Obs(
            gap_dist=self._distance_to_next(Tile.GAP),
            enemy_dist=self._distance_to_next(Tile.ENEMY),
            projectile_dist=self._distance_to_next_projectile(),
            on_ground=(self.air_time == 0),
            air_time=self.air_time,
        )
