import sys
import pygame
import argparse

from .logger import JsonlLogger, obs_to_dict

from game.env import PlatformerEnv, Action, Tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default=None,
                        help="Path to JSONL log file (e.g., data/run1.jsonl)")
    args = parser.parse_args()

    logger = JsonlLogger(args.log, flush_every=10) if args.log else None

    pygame.init()
    WIDTH, HEIGHT = 1000, 300
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Platformer RL-ILP (Visual Debug)")
    clock = pygame.time.Clock()

    # --- Environment ---
    env = PlatformerEnv(seed=0, length=300, lookahead=10,
                        p_gap=0.08, p_enemy=0.08)
    obs = env.reset()

    # --- Rendering params ---
    tile_size = 32
    ground_y = HEIGHT - 60  # baseline
    player_x_px = 200       # fixed on screen, world scrolls
    jump_height = 60        # pixels

    # How many tiles to draw across screen
    tiles_on_screen = WIDTH // tile_size + 2

    # Control
    action = Action.DO_NOTHING
    running = True

    font = pygame.font.SysFont(None, 22)

    try:
        while running:
            clock.tick(30)  # FPS

            # --- Handle events ---
            action = Action.DO_NOTHING
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False
            if keys[pygame.K_SPACE]:
                action = Action.JUMP
            elif keys[pygame.K_a]:
                action = Action.ATTACK

            # --- Step env ---
            state_before = obs  # save for logging
            step = env.step(action)
            obs = step.obs

            # --- Log ---
            if logger:
                logger.log({
                    "mode": "play",
                    "state_before": obs_to_dict(state_before),
                    "action": action.value,
                    "reward": step.reward,
                    "event": step.event.value,
                    "done": step.done,
                    "info": step.info,
                })

            # --- Reset on death / end ---
            if step.done:
                obs = env.reset()

            # --- Draw ---
            screen.fill((20, 20, 20))

            # Determine which tile index aligns with player screen position.
            # Player is at env.player_x, and rendered at player_x_px.
            # So tile at env.player_x is at screen position player_x_px.
            start_tile = env.player_x - (player_x_px // tile_size)
            if start_tile < 0:
                start_tile = 0

            # Draw tiles
            for i in range(tiles_on_screen):
                tile_index = start_tile + i
                x = i * tile_size

                if 0 <= tile_index < len(env.track):
                    tile = env.track[tile_index]
                else:
                    tile = Tile.GROUND

                # ground baseline rectangle
                ground_rect = pygame.Rect(x, ground_y, tile_size, 60)

                if tile == Tile.GROUND:
                    pygame.draw.rect(screen, (70, 50, 30), ground_rect)
                elif tile == Tile.GAP:
                    # draw nothing for gap (leave background), maybe border
                    pygame.draw.rect(screen, (40, 40, 40), ground_rect, 1)
                elif tile == Tile.ENEMY:
                    pygame.draw.rect(screen, (70, 50, 30), ground_rect)
                    enemy_rect = pygame.Rect(
                        x + 6, ground_y - 26, tile_size - 12, 26)
                    pygame.draw.rect(screen, (200, 50, 50), enemy_rect)

            # Draw player
            player_on_ground = obs.on_ground
            y = ground_y - 30
            if not player_on_ground:
                # simple jump animation: lift while in air
                y = (ground_y - 30) - jump_height

            player_rect = pygame.Rect(player_x_px + 6, y, tile_size - 12, 30)
            pygame.draw.rect(screen, (70, 130, 255), player_rect)

            # HUD
            hud_lines = [
                f"x={env.player_x}  t={env.t}",
                f"enemy_dist={obs.enemy_dist}  gap_dist={obs.gap_dist}  air_time={obs.air_time}",
                f"last_event={env.last_event.value}",
                "Controls: SPACE=jump, A=attack, ESC=quit",
                f"Logging: {'ON -> ' + args.log if logger else 'OFF'}",
            ]
            for j, text in enumerate(hud_lines):
                img = font.render(text, True, (230, 230, 230))
                screen.blit(img, (10, 10 + 20 * j))

            pygame.display.flip()

    finally:
        if logger:
            logger.close()
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    main()
