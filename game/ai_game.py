import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np


# ----- SETUP -----
pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    """
    Represent direction snake is moving using enumeration
    """
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# represent a block
Block = namedtuple('Block', 'x, y')

# colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
LIGHT_BLUE = (0, 0, 255)
DARK_BLUE = (0, 100, 255)
BLACK = (0, 0, 0)

# game constants
BLOCK_SIZE = 20
SPEED = 40

# ----- GAME BEHAVIOR -----
class SnakeGameAI:
    """
    Represent a game of Snake
    """
    def __init__(self, w=640, h=480):
        # display
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        # game state
        self.reset()

    def reset(self):
        """
        Reset game state
        :return: None
        """
        self.direction = Direction.RIGHT
        self.head = Block(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Block(self.head.x - BLOCK_SIZE, self.head.y),
                      Block(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        # ensure AI moves in timely manner
        self.frame_iteration = 0

    def _place_food(self):
        """
        Place food block randomly on display
        :return: None
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Block(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """
        Update game state based upon AI action
        :return: <tuple> (<bool> whether game is over, <int> current score)
        """
        self.frame_iteration += 1
        # user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # move
        self._move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        # collision or time exceeded
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        # place new food or simply move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        # update display
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """
        Check whether there is a collision
        :return: True if collision, False if not
        """
        if pt is None:
            pt = self.head
        # hit boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hit self
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        """
        Update display with current game state
        :return: None
        """
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, LIGHT_BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, DARK_BLUE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        Move the snake based upon user input
        :param action: <array> relative direction to move [straight, right, left]
        :return: None
        """
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clockwise.index(self.direction)
        # straight (no change)
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clockwise[index]
        # right turn (R -> D -> L -> U)
        elif np.array_equal(action, [0, 1, 0]):
            next_index = (index + 1) % 4
            new_direction = clockwise[next_index]
        # left turn (R -> U -> L -> D)
        else: # [0, 0, 1]
            prev_index = (index - 1) % 4
            new_direction = clockwise[prev_index]
        self.direction = new_direction
        # update snake state
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Block(x, y)
