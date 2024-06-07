import gymnasium as gym
from itertools import count
import torch
from params import DEVICE, MULTIFRAME


class Environment():
    def __init__(self, env_gym: gym.Env) -> None:
        self.env = env_gym

    def run_episode(self, agent) -> int:
        """
        Runs a single episode of the environment using the provided agent.
        Store transition in memory and move to the next state.
        Performance optimization and update target.

        CHECK VALUE OF MULTIFRAME global variable. Has to be set to 1 for this method to be called correctly

        Args:
            agent (_type_): Component that makes the decision of what action to take
        """
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        for t in count():
            action = agent.select_action(self.env.action_space, state) # , self.env.get_wrapper_attr('desc')
            observation, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward], device=DEVICE)
            done = terminated or truncated

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Store the transition in memory
            reset = agent.update_memory(state, action, next_state, reward) # , self.env.get_wrapper_attr('desc')
            agent.logging()

            if reset is not None and reset:
                break

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            agent.soft_update_agent()

            if done:
                break

        return t

    def run_episode_memory(self, agent) -> int:
        """
        Runs a single episode of the environment using the provided agent. With a multiframed input
        Store transition in memory and move to the next state.
        Performance optimization and update target.

        Args:
            agent (_type_): Component that makes the decision of what action to take
        """
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        batch = []
        for _ in range(MULTIFRAME): #Creates a batch of mutiple frames
            batch.append(state)


        for t in count():

            state = torch.cat(batch, -1).to(DEVICE) #Transforms batch into right format
            action = agent.select_action(self.env.action_space, state) # , self.env.get_wrapper_attr('desc')
            observation, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward], device=DEVICE)
            done = terminated or truncated

            if done:
                next_state = None
            else: #Creates the next state by appending the new observation, popping the first one from the list
                # and then transforming it to the right format
                batch.append(torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                batch.pop(0)
                next_state = torch.cat(batch,-1)

            # Store the transition in memory
            reset = agent.update_memory(state, action, next_state, reward) # , self.env.get_wrapper_attr('desc')
            if reset is not None :
                if reset:
                    break
            agent.logging()

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            agent.soft_update_agent()

            if done:
                break

        return t
