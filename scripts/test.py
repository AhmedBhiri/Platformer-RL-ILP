from game.env import PlatformerEnv, Action

env = PlatformerEnv(seed=0)
obs = env.reset()
done = False

while not done:
    # random agent for now
    action = Action.DO_NOTHING
    if obs.enemy_dist == 1:
        action = Action.ATTACK
    elif obs.gap_dist == 1:
        action = Action.JUMP

    step = env.step(action)
    obs = step.obs
    done = step.done
    print(step.info, step.event, step.reward)

print("Finished")
