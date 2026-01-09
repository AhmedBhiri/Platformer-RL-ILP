import sys
import pygame

from game.env import PlatformerEnv, Action, Tile


def main():
    pygame.init()
    WIDTH, HEIGHT = 1000, 300
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Platformer RL-ILP (Slow Visual Debug)")
    clock = pygame.time.Clock()

    # --- Environment ---
    env = PlatformerEnv(seed=0, length=300, lookahead=10,
                        p_gap=0.08, p_enemy=0.08, p_shoot=0.15)
    obs = env.reset()

    # --- Rendering params ---
    tile_size = 32
    ground_y = HEIGHT - 60
    player_x_px = 200
    jump_height = 60

    tiles_on_screen = WIDTH // tile_size + 2

    # --- Slowdown parameters ---
    FPS = 15
    STEP_EVERY_N_FRAMES = 1
    frame_counter = 0

    font = pygame.font.SysFont(None, 22)
    running = True
    action = Action.DO_NOTHING

    # Jump should be edge-triggered (press, not hold)
    prev_space = False

    while running:
        clock.tick(FPS)
        frame_counter += 1

        # --- Handle input ---
        action = Action.DO_NOTHING
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        space = keys[pygame.K_SPACE]
        attack = keys[pygame.K_a]
        duck = keys[pygame.K_DOWN] or keys[pygame.K_s]

        # Priority: jump (press) > duck (hold) > attack (hold) > do_nothing
        if space and not prev_space:
            action = Action.JUMP
        elif duck:
            action = Action.DUCK
        elif attack:
            action = Action.ATTACK

        prev_space = space

        # --- Step env ---
        if frame_counter % STEP_EVERY_N_FRAMES == 0:
            step = env.step(action)
            obs = step.obs
            if step.done:
                obs = env.reset()

        # --- Draw ---
        screen.fill((20, 20, 20))

        start_tile = env.player_x - (player_x_px // tile_size)
        if start_tile < 0:
            start_tile = 0

        # Draw tiles + enemies
        for i in range(tiles_on_screen):
            tile_index = start_tile + i
            x = i * tile_size

            tile = env.track[tile_index] if 0 <= tile_index < len(
                env.track) else Tile.GROUND
            ground_rect = pygame.Rect(x, ground_y, tile_size, 60)

            if tile == Tile.GROUND:
                pygame.draw.rect(screen, (70, 50, 30), ground_rect)
            elif tile == Tile.GAP:
                pygame.draw.rect(screen, (40, 40, 40), ground_rect, 1)
            elif tile == Tile.ENEMY:
                pygame.draw.rect(screen, (70, 50, 30), ground_rect)
                enemy_rect = pygame.Rect(
                    x + 6, ground_y - 26, tile_size - 12, 26)
                pygame.draw.rect(screen, (200, 50, 50), enemy_rect)

        # Draw projectiles (yellow squares)
        for p in getattr(env, "projectiles", []):
            if start_tile <= p < start_tile + tiles_on_screen:
                i = p - start_tile
                x = i * tile_size
                proj_rect = pygame.Rect(x + 12, ground_y - 18, 8, 8)
                pygame.draw.rect(screen, (240, 220, 80), proj_rect)

        # Player (duck/jump visuals)
        if not obs.on_ground:
            # jumping: move up
            y = (ground_y - 30) - jump_height
            player_h = 30
        elif duck:
            # ducking: shorter
            y = ground_y - 18
            player_h = 18
        else:
            # standing
            y = ground_y - 30
            player_h = 30

        player_rect = pygame.Rect(player_x_px + 6, y, tile_size - 12, player_h)
        pygame.draw.rect(screen, (70, 130, 255), player_rect)

        # HUD
        hud_lines = [
            "SLOW MODE",
            f"x={env.player_x}  t={env.t}",
            f"enemy_dist={obs.enemy_dist}  gap_dist={obs.gap_dist}  proj_dist={obs.projectile_dist}  air_time={obs.air_time}",
            f"projectiles={len(getattr(env, 'projectiles', []))}",
            f"last_event={env.last_event.value}",
            "Controls: SPACE=jump (press), DOWN or S=duck (hold), A=attack (hold), ESC=quit",
        ]
        for j, text in enumerate(hud_lines):
            img = font.render(text, True, (230, 230, 230))
            screen.blit(img, (10, 10 + 20 * j))

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
