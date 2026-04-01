import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import imageio
from PIL import Image, ImageDraw, ImageFont

class MountainCarAgent:
    def __init__(self, learning_rate=0.2, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, discount=0.99):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount = discount

        self.Q = defaultdict(lambda: np.zeros(3))  # 3 actions: left, neutral, right

    def discretize_state(self, state):
        position, velocity = state
        pos_bins = np.linspace(-1.2, 0.6, 20)
        vel_bins = np.linspace(-0.07, 0.07, 20)
        pos_idx = np.digitize(position, pos_bins) - 1
        vel_idx = np.digitize(velocity, vel_bins) - 1

        pos_idx = max(0, min(19, pos_idx))
        vel_idx = max(0, min(19, vel_idx))

        return (pos_idx, vel_idx)

    def choose_action(self, state, training=True):
        discretize_state = self.discretize_state(state)
        if training and np.random.random() < self.epsilon:
            return np.random.randint(3)  # Explore
        else:
            return np.argmax(self.Q[discretize_state])  # Exploit

    def learn(self, state, action, reward, next_state, done):
        discretize_state = self.discretize_state(state)
        discretize_next_state = self.discretize_state(next_state)

        position, velocity = next_state
        # Sparse values (-1 per step and +1 for reaching the goal)
        # position-based bonus r'(new reward) = r + <some_value> * current_position
        new_reward = reward + position * 10  # parameter which we can tune overtime

        current_q = self.Q[discretize_state][action]
        if done:
            next_max_q = 0
        else:
            next_max_q = max(self.Q[discretize_next_state])

        new_q = current_q + self.lr * (new_reward + self.discount * next_max_q - current_q)
        self.Q[discretize_state][action] = new_q

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent():
    env = gym.make('MountainCar-v0')
    agent = MountainCarAgent()
    successful = 0

    for episode in range(20000):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Unpack if reset returns (obs, info)

        done = False
        steps = 0

        while not done and steps < 200:
            action = agent.choose_action(state, training=True)
            result = env.step(action)
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            steps += 1

            # On day one you have acquired some knowledge (explored) -> use knowledge we move to next day

            if next_state[0] >= 0.5:  # Goal position reached
                successful += 1
                break

        if (episode + 1) % 2000 == 0:
            success_rate = (successful / (episode + 1)) * 100
            print(f"Episode: {episode + 1}/20000 | Successes: {successful} ({success_rate:.2f}%)")

    env.close()

    return agent


def record_episode_frames(env, agent=None, max_steps=200):
    frames = []
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # Unpack if reset returns (obs, info)

    done = False
    steps = 0
    max_position = -999
    reached_goal = False

    while not done and steps < max_steps:

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        max_position = max(max_position, state[0])

        if agent is None:
            action = env.action_space.sample()  # Random action
        else:
            action = agent.choose_action(state, training=False)  # Use trained agent

        result = env.step(action)
        state, reward, terminated, truncated, info = result
        done = terminated or truncated

        steps += 1

        if state[0] >= 0.5:  # Goal position reached
            reached_goal = True
            for _ in range(10):  # Add extra frames to show the success
                frame = env.render()
                if frame is not None:
                    for _ in range(3):  # Show the success for a few frames
                        frames.append(frame)
            break

    return frames, steps, max_position, reached_goal


def create_gif_with_text(frames, filename, text, steps, max_pos, reached_goal):
    processed_frames = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        status = "Goal Reached!" if reached_goal and i >= len(frames) - 30 else f"Steps: {min((i//3) + 1, steps)} | Max Pos: {max_pos:.3f}"
        title = f"{text}\n{status}\nMax Position: {max_pos:.3f}"

        text_color = (0, 255, 0) if reached_goal and i >= len(frames) - 30 else (0, 0, 0)
        draw.text((10, 10), title, fill=text_color, font=font)
        processed_frames.append(np.array(img))

    imageio.mimwrite(filename, processed_frames, duration=20, loop=0)


trained_agent = train_agent()

env_untrained = gym.make('MountainCar-v0', render_mode='rgb_array')
frames_untrained, steps_untrained, max_position_untrained, reached_goal_untrained = record_episode_frames(env_untrained)
env_untrained.close()

env_trained = gym.make('MountainCar-v0', render_mode='rgb_array')
frames_trained, steps_trained, max_position_trained, reached_goal_trained = record_episode_frames(env_trained, agent=trained_agent)
env_trained.close()

create_gif_with_text(frames_untrained, "mountain_car_untrained.gif", "Mountain Car - Untrained Agent", steps_untrained, max_position_untrained, reached_goal_untrained)
create_gif_with_text(frames_trained, "mountain_car_trained.gif", "Mountain Car - Trained Agent", steps_trained, max_position_trained, reached_goal_trained)
