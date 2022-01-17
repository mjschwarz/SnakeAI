import torch
import random
import numpy as np
from collections import deque
from game.ai_game import SnakeGameAI, Direction, Block
from model.model import Linear_QNet, QTrainer
from helper import plot


# training constants
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent():
    """
    Represent an AI agent
    """
    def __init__(self):
        """
        Initialize an agent
        """
        self.num_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """
        Get the current danger, movement, and food states
        :param game: <SnakeGameAI> game being played
        :return: <np.array> states as [DNG_S, DNG_R, DNG_D, DIR_L, DIR_R, DIR_U, DIR_D, F_L, F_R, F_U, F_D]
        """
        head = game.snake[0]
        # danger blocks
        block_l = Block(head.x - 20, head.y)
        block_r = Block(head.x + 20, head.y)
        block_u = Block(head.x, head.y - 20)
        block_d = Block(head.x, head.y + 20)
        # directions
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        # danger, movement, food states
        state = [
            # danger straight
            (dir_l and game.is_collision(block_l)) or
            (dir_r and game.is_collision(block_r)) or
            (dir_u and game.is_collision(block_u)) or
            (dir_d and game.is_collision(block_d)),
            # danger right
            (dir_l and game.is_collision(block_u)) or
            (dir_r and game.is_collision(block_d)) or
            (dir_u and game.is_collision(block_r)) or
            (dir_d and game.is_collision(block_l)),
            # danger left
            (dir_l and game.is_collision(block_d)) or
            (dir_r and game.is_collision(block_u)) or
            (dir_u and game.is_collision(block_l)) or
            (dir_d and game.is_collision(block_r)),
            # movement direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # relative food direction
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.x > game.head.x,  # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        """
        Store the current game state in memory
        :param state: <np.array> current danger, movement, and food states
        :param action: <array> direction to move [straight, left, right]
        :param reward: <int> reward amount
        :param next_state: <np.array> next danger, movement, and food states
        :param game_over: <bool> whether game is finished
        :return: None
        """
        # automatically pops left if max memory reached
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        """
        Train the long-term/replay/experience memory
        :return: None
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        """
        Train the short-term memory
        :param state: <np.array> current danger, movement, and food states
        :param action: <array> direction to move [straight, left, right]
        :param reward: <int> reward amount
        :param next_state: <np.array> next danger, movement, and food states
        :param game_over: <bool> whether game is finished
        :return: None
        """
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state, using_cache):
        """
        Get the AI action (movement direction)
        :param state: <np.array> current danger, movement, and food states
        :param using_cache: <bool> whether the agent is using cached model state or not
        :return: <array> direction to move [straight, left, right]
        """
        # random moves: tradeoff between exploration and exploitation
        self.epsilon = 80 - self.num_games
        final_move = [0, 0, 0]
        # random move
        if not using_cache and random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        # predicted move
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # execute forward function
            move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move


def train():
    """
    Train the agent
    :return: None
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    high_score = 0
    agent = Agent()
    using_cache = agent.model.load()
    game = SnakeGameAI()
    # training loop
    while True:
        # get old state
        state_old = agent.get_state(game)
        # get move
        final_move = agent.get_action(state_old, using_cache)
        # make move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)
        # store state in memory
        agent.remember(state_old, final_move, reward, state_new, game_over)
        if game_over:
            # train long memory
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()
            if score > high_score:
                high_score = score
                agent.model.save()
            print('Game', agent.num_games, 'Score:', score, 'High Score', high_score)
            # plot result
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


# ----- TRAIN MODEL -----
if __name__ == '__main__':
    train()