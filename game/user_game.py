import pygame
import random
from enum import Enum
from collections import namedtuple


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
SPEED = 20

# ----- GAME BEHAVIOR -----
class SnakeGame:
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
        self.direction = Direction.RIGHT
        self.head = Block(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Block(self.head.x - BLOCK_SIZE, self.head.y),
                      Block(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()

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

    def play_step(self):
        """
        Update game state based upon user input
        :return: <tuple> (<bool> whether game is over, <int> current score)
        """
        # user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        # move
        self._move(self.direction)
        self.snake.insert(0, self.head)
        game_over = False
        # collision
        if self._is_collision():
            game_over = True
            return game_over, self.score
        # place new food or simply move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        # update display
        self._update_ui()
        self.clock.tick(SPEED)
        return game_over, self.score

    def _is_collision(self):
        """
        Check whether there is a collision
        :return: True if collision, False if not
        """
        # hit boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # hit self
        if self.head in self.snake[1:]:
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

    def _move(self, direction):
        """
        Move the snake based upon user input
        :param direction: <int> direction to move
        :return: None
        """
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Block(x, y)


# ----- GAME LOOP -----
if __name__ == '__main__':
    game = SnakeGame()
    while True:
        game_over, score = game.play_step()
        if game_over == True:
            break
    print('Final Score', score)
    pygame.quit()