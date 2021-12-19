from mlagents_envs.environment import UnityEnvironment

env = UnityEnvironment(file_name='./env/linux/GcMaze.x86_64', no_graphics=True)
env.reset()

behavior_name = list(env.behavior_specs._dict.keys())[0]
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name]
decision_steps, terminal_steps = env.get_steps(behavior_name)
tracked_agent = -1

done = False
episode_rewards = 0

done_num = 0
step = 0

alive_agent = 100

for _ in range(10000):
    # env.set_actions(behavior_name, spec.create_random_action(len(decision_steps)))
    env.set_actions(behavior_name, spec.action_spec.random_action(len(decision_steps)))
    env.step()
    step += 1

    # if tracked_agent == -1 and len(decision_steps) >= 1:
    #     tracked_agent = decision_steps.agent_id[0]

    decision_steps, terminal_steps = env.get_steps(behavior_name)
    
    print(f"step = {step} agent = {len(decision_steps.agent_id)} done = {len(terminal_steps.agent_id)} alive = {alive_agent}")

    if terminal_steps.agent_id.shape[0] != 0:
        print(f"{terminal_steps.agent_id} is done, {decision_steps.agent_id}") 
        done_num += len(decision_steps.agent_id)
        alive_agent -= len(decision_steps.agent_id)
    # if done_num >= 4:
    #     env.reset()
    #     done_num = 0
    #     step = 0
    # elif decision_steps.agent_id.shape[0] + terminal_steps.agent_id.shape[0] != 4 :
    #     print("Dont get four obs")
    # if tracked_agent in decision_steps: # The agent requested a decision
    #     episode_rewards += decision_steps[tracked_agent].reward
    # if tracked_agent in terminal_steps: # The agent terminated its episode
    #     episode_rewards += terminal_steps[tracked_agent].reward
    #     done = True

env.close()
print("Closed environment")
