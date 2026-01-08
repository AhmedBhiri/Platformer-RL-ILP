import sys
import pygame

from game.env import PlatformerEnv, Action, Tile


def choose_action(obs):
    """
    Policy to visualize: ATTACK when enemy is attackable, JUMP when pit near, else DO_NOTHING.
    """

    # 1) Attack has the tightest timing window -> prioritize it
    if obs.enemy_dist is not None and obs.enemy_dist <= 1:
        return Action.ATTACK

    # 2) Jump for pits
    if obs.gap_dist is not None and obs.gap_dist <= 3:
        return Action.JUMP

    # 3) Default
    return Action.DO_NOTHING


def main():
    pygame.init()
    WIDTH, HEIGHT = 1000, 300
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Platformer RL-ILP (Learned Policy)")
    clock = pygame.time.Clock()

    env = PlatformerEnv(seed=0, length=300, lookahead=10)
    obs = env.reset()

    tile_size = 32
    ground_y = HEIGHT - 60
    player_x_px = 200
    jump_height = 60
    tiles_on_screen = WIDTH // tile_size + 2

    font = pygame.font.SysFont(None, 22)
    running = True

    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = choose_action(obs)
        step = env.step(action)
        obs = step.obs

        if step.done:
            obs = env.reset()

        # --- Draw ---
        screen.fill((20, 20, 20))

        start_tile = env.player_x - (player_x_px // tile_size)
        if start_tile < 0:
            start_tile = 0

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

        y = ground_y - 30
        if not obs.on_ground:
            y -= jump_height

        player_rect = pygame.Rect(player_x_px + 6, y, tile_size - 12, 30)
        pygame.draw.rect(screen, (70, 130, 255), player_rect)

        hud = [
            f"action={action.value}",
            f"gap_dist={obs.gap_dist}",
            f"on_ground={obs.on_ground}",
        ]
        for i, text in enumerate(hud):
            img = font.render(text, True, (230, 230, 230))
            screen.blit(img, (10, 10 + 20 * i))

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
