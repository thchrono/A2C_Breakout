import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.init import kaiming_uniform_, orthogonal_
from torch.nn.utils import clip_grad_norm_
import ale_py # without this import we can't load any atari game
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import gym

class ActorCritic(nn.Module):
    def __init__(self, input_dim, conv1_filter, conv2_filter, conv3_filter, hidden_dim, output_dim):
        super(ActorCritic, self).__init__()

        # input shape [Batch_size,4,84,84]
        # input_dim shape [4,84,84]
        # self.conv1 shape [Batch_size,conv1_filter,20,20]
        self.conv1 = nn.Conv2d(in_channels=input_dim[0], out_channels=conv1_filter, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        # self.conv2 shape [Batch_size,conv2_filter,9,9]
        self.conv2 = nn.Conv2d(in_channels=conv1_filter, out_channels=conv2_filter, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        # self.conv3 shape [Batch_size,conv2_filter,7,7]
        self.conv3 = nn.Conv2d(in_channels=conv2_filter, out_channels=conv3_filter, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=conv3_filter*7*7, out_features=hidden_dim)
        self.relu4 = nn.ReLU()

        # actor
        self.actor = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        # critic
        self.critic = nn.Linear(in_features=hidden_dim, out_features=1)

        # initialize weights
        self._weight_init()

    def _weight_init(self):
        with torch.no_grad():
            # He initialization for all layers with relu activation
            kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
            kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
            kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
            kaiming_uniform_(self.fc1.weight, nonlinearity='relu')

            # zero biases
            self.conv1.bias.data.zero_()
            self.conv2.bias.data.zero_()
            self.conv3.bias.data.zero_()
            self.fc1.bias.data.zero_()

            # orthogonal init for actor and critic
            orthogonal_(self.actor.weight, gain=0.01)
            orthogonal_(self.critic.weight)

            # zero biases for actor and critic
            self.actor.bias.data.zero_()
            self.critic.bias.data.zero_()

            print("Weight initialization complete!")

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))

        # actor
        policy = self.actor(x)
        # critic
        value = self.critic(x)

        return policy, value

class A2C_Network:
    def __init__(self,env_id,gamma,total_updates,learning_rate,input_dim,
                 conv1_filter,conv2_filter,conv3_filter,hidden_dim,
                 entropy_coef,critic_coef, clip_grad):
        self.env_id = env_id
        self.env = make_atari_env(self.env_id, n_envs=1, seed=0)
        self.gamma = gamma
        self.total_updates = total_updates
        self.training_rewards = []
        self.training_losses = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", self.device)

        #-------------------- Actor-Critic --------------------
        self.state_dim = input_dim
        self.conv1_filter = conv1_filter
        self.conv2_filter = conv2_filter
        self.conv3_filter = conv3_filter
        self.hidden_dim = hidden_dim
        self.num_actions = len(self.env.get_attr('_action_set')[0])

        self.model = (ActorCritic(self.state_dim,self.conv1_filter,self.conv2_filter,
                                  self.conv3_filter,self.hidden_dim,self.num_actions)
                      ).to(self.device)

        #---------------- Optimizer ----------------
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.99, eps=1e-5)

        # linear learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.14, total_iters=self.total_updates)
        self.clip_grad = clip_grad

        #---------------- Loss coefficients ----------------
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef

    def select_action(self, states):
        # states: [num_envs,filters,heights,width]
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        action_logits, values = self.model(states)

        # calculate distribution
        distribution = Categorical(logits=action_logits)
        # get an action based on the probabilities of the distribution
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        entropies = distribution.entropy()

        return actions.cpu().numpy(), values, log_probs, entropies


    def training(self,num_envs,n_steps, seed):
        '''
            Trains the A2C agent using n-step rollouts across multiple environments.

            Inputs:
                num_envs (int): Number of parallel environments.
                n_steps (int): Number of steps of rollout per environment before training.
        '''

        self.model.train()

        # create envs for parallel training
        # make_atari_env clips reward by default to [-1,1]
        # can be deactivated with wrapper_kwargs={'clip_reward': False}
        # but showed worse performance overall
        envs = make_atari_env(self.env_id,
                              n_envs=num_envs,
                              seed=seed)

        envs = VecFrameStack(envs, n_stack=4)
        states = envs.reset()
        # print(states.shape)
        states = np.moveaxis(states, -1, 1)
        # print(states.shape)
        states = torch.from_numpy(states).float().to(self.device)

        for update in range(self.total_updates):
            # buffers
            state_buf, action_buf, reward_buf, done_buf = [], [], [], []
            values_buf, log_probs_buf, entropies_buf = [], [], []

            # rollout
            for _ in range(n_steps):
                # select action and get new state
                actions, values, log_probs, entropies = self.select_action(states=states)
                new_states, rewards, done, _ = envs.step(actions)
                new_states = np.moveaxis(new_states, -1, 1)

                # convert action from numpy to tensor
                actions = torch.tensor(actions).to(self.device)
                rewards = torch.tensor(rewards).to(self.device)
                done = torch.from_numpy(done).float().to(self.device)

                # update buffers
                state_buf.append(states)
                action_buf.append(actions)
                reward_buf.append(rewards)
                done_buf.append(done)
                values_buf.append(values.squeeze(-1))
                log_probs_buf.append(log_probs)
                entropies_buf.append(entropies)

                # update states
                states = torch.from_numpy(new_states).float().to(self.device)

            # we need the value (V(S_{t+n})) to compute the TD(n) error
            # n = n_steps so we need to calculate the value of the next
            # rollout
            with torch.no_grad():
                _, next_value = self.model(states)
                next_value = next_value.squeeze(-1)

            # compute returns
            returns = []
            R = next_value
            for step in reversed(range(n_steps)):
                R = reward_buf[step] + self.gamma * R * (1 - done_buf[step])
                returns.append(R)

            # in the end of the loop, returns = [Rn, ..., R1, R0]
            # need to reverse so returns = [R0, R1, ..., Rn]
            returns.reverse()
            returns = torch.stack(returns)  # shape [n_steps, num_envs]

            # stack rollout buffers (shape [n_steps, num_envs])
            values = torch.stack(values_buf)
            log_probs = torch.stack(log_probs_buf)
            entropies = torch.stack(entropies_buf)

            # flatten (shape [n_steps * num_envs])
            returns = returns.view(-1)
            values = values.view(-1)
            log_probs = log_probs.view(-1)
            entropies = entropies.view(-1)

            # compute advantages
            advantages = returns - values

            # compute losses
            # detach to not let gradients flow
            # through critic when updating the actor
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()
            entropy_loss = entropies.mean()

            loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy_loss

            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
            self.optimizer.step()
            self.scheduler.step()


            self.training_losses.append(loss.item())
            update_rewards = torch.stack(reward_buf).sum(dim=0)  # shape [num_envs]
            mean_reward = update_rewards.mean().item()
            self.training_rewards.append(mean_reward)

            if update % 100 == 0:
                print(
                    f"Update {update}, Avg reward: {mean_reward:.4f}, Actor: {actor_loss.item():.4f}, "
                    f"Critic: {critic_loss.item():.4f}, Entropy: {entropy_loss.item():.4f}, Loss: {loss.item():.4f}")

        # smooth the rewards and loss every 100 updates
        window_size = 100
        averaged_rewards = []
        averaged_losses = []
        for i in range(0, len(self.training_rewards), window_size):
            averaged_rewards.append(np.mean(self.training_rewards[i:i + window_size]))
            averaged_losses.append(np.mean(self.training_losses[i:i + window_size]))

        averaged_updates = list(range(0, len(self.training_rewards), window_size))

        plt.figure(figsize=(12, 6))
        plt.plot(averaged_updates, averaged_rewards, label="Average Reward (per 100 updates)", linewidth=2)
        plt.xlabel("Update")
        plt.ylabel("Average Reward")
        plt.title("A2C Performance Over Updates (smoothed)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("a2c_training_rewards.png")
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(averaged_updates, averaged_losses, label="Average Loss (per 100 updates)", linewidth=2)
        plt.xlabel("Update")
        plt.ylabel("Loss")
        plt.title("A2C Loss Over Updates (smoothed)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("a2c_training_loss.png")
        plt.show()

        torch.save(self.model.state_dict(), "a2c_model.pth")
        print("Model saved to a2c_model.pth")

    def save_policy_video(self, output_filename="a2c_policy.mp4", episodes=3):
        """
        Runs the learned A2C policy and saves the simulation as an MP4 video.

        Args:
            output_filename (str): File path for saving the video.
            episodes (int): Number of episodes to record.
        """

        # load weights
        self.model.load_state_dict(torch.load("a2c_model.pth"))
        self.model.to(self.device)
        self.model.eval()

        # create env
        env = make_atari_env(self.env_id,
                             n_envs=1,
                             seed=np.random.randint(1e4),
                             wrapper_kwargs={'clip_reward': False})

        env = VecFrameStack(env, n_stack=4)

        frames = []
        rewards_per_episode = []

        for ep in range(episodes):
            state = env.reset()
            state = np.moveaxis(state, -1, 1)
            total_reward = 0
            done = False

            # initial frame
            frame = env.render()
            frames.append(frame)

            while not done:
                # get action
                action, _, _, _ = self.select_action(state)
                # step env
                new_state, reward, done, _ = env.step(action)
                new_state = np.moveaxis(new_state, -1, 1)
                total_reward += reward[0]

                frame = env.render()
                frames.append(frame)

                state = new_state

            rewards_per_episode.append(total_reward)
            print(f"Episode {ep}: Total Reward = {int(rewards_per_episode[ep])}")

            # add some sleep time for the last frame or each episode
            for _ in range(15):
                frames.append(frame)

        env.close()

        # create animation
        fig, ax = plt.subplots()
        img = ax.imshow(frames[0], cmap='gray')
        plt.axis('off')

        def animate(i):
            img.set_array(frames[i])
            return[img]

        ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
        ani.save(output_filename, writer='ffmpeg', fps=30, dpi=100)
        print(f"Saved policy video to '{output_filename}'")