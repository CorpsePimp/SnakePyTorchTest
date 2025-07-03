# -*- coding: utf-8 -*-
"""
Snake AI (DQN + PyTorch + Pygame)
Полная реализация игры "Змейка" с самообучающимся ИИ-агентом.
Автор: ChatGPT, 2025
"""

# ---------------- 1. Импорты ----------------
import pygame
import random
from enum import Enum
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os

# ---------------- 2. Константы и настройки ----------------
BLOCK_SIZE = 20
GRID_SIZE = 20
SPEEDS = [30, 60, 120, 500]    # Возможные скорости (кадр/сек)
BUFFER_SIZE = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9

MODEL_PATH = "snake_dqn_model.pth"

# Цвета
WHITE = (255,255,255)
RED = (200,0,0)
GREEN = (0,200,0)
BLACK = (0,0,0)
GRAY = (128,128,128)

pygame.init()
font = pygame.font.SysFont('Arial', 25)

# ---------------- 3. Класс игры Snake ----------------

class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

Point = namedtuple('Point', 'x y')

class SnakeGame:
    def __init__(self, w=GRID_SIZE*BLOCK_SIZE, h=GRID_SIZE*BLOCK_SIZE):
        self.w = w
        self.h = h
        self.reset()
    
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w//2, self.h//2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - 2*BLOCK_SIZE, self.head.y)]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0
        self.prev_dist = self.get_food_distance()
    
    def place_food(self):
        while True:
            x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break
    
    def play_step(self, action, human_mode=False):
        self.frame_iteration += 1
        # 1. Движение
        self.move(action)
        self.snake.insert(0, self.head)
        
        # 2. Проверка столкновений
        game_over = False
        reward = -1 # Штраф за каждый шаг (поощряет быстрее искать еду)
        if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 3. Проверка еды
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()
        
        # 4. Доп. награда за движение к еде или от нее
        new_dist = self.get_food_distance()
        if new_dist < self.prev_dist:
            reward += 1
        elif new_dist > self.prev_dist:
            reward -= 1
        self.prev_dist = new_dist
        
        # 5. Принудительный game over (замкнутая траектория)
        if self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
        
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Столкновение со стеной
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Столкновение с собой
        if pt in self.snake[1:]:
            return True
        return False
    
    def move(self, action):
        # action: [straight, right, left] => 0, 1, 2
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1,0,0]):
            new_dir = clock_wise[idx]      # прямо
        elif np.array_equal(action, [0,1,0]):
            new_dir = clock_wise[(idx+1)%4] # вправо
        else: # [0,0,1]
            new_dir = clock_wise[(idx-1)%4] # влево
        self.direction = new_dir
        
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        self.head = Point(x, y)
    
    def get_food_distance(self):
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
    
    def get_state(self):
        # 11 признаков: опасность впереди/влево/вправо, текущее направление (4), положение еды (4)
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        point_l = Point(self.head.x - BLOCK_SIZE, self.head.y)
        point_r = Point(self.head.x + BLOCK_SIZE, self.head.y)
        point_u = Point(self.head.x, self.head.y - BLOCK_SIZE)
        point_d = Point(self.head.x, self.head.y + BLOCK_SIZE)
        
        # Опасности
        danger_straight = (
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d))
        )
        danger_right = (
            (dir_u and self.is_collision(point_r)) or
            (dir_r and self.is_collision(point_d)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u))
        )
        danger_left = (
            (dir_d and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_d)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u))
        )
        
        # Еда (расположение относительно головы)
        food_left = self.food.x < self.head.x
        food_right = self.food.x > self.head.x
        food_up = self.food.y < self.head.y
        food_down = self.food.y > self.head.y
        
        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down)
        ]
        return np.array(state, dtype=int)
    
    def render(self, display, record=0, avg_score=0, speed=0, game_idx=0, mode='train'):
        display.fill(BLACK)
        # Рисуем змейку
        for pt in self.snake:
            pygame.draw.rect(display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        # Еда
        pygame.draw.rect(display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # Счетчик
        info = f"Игра: {game_idx} | Рекорд: {record} | Среднее: {avg_score:.2f} | Скорость: {speed} | Режим: {mode}"
        text = font.render(info, True, WHITE)
        display.blit(text, [10, 10])
        pygame.display.flip()

# ---------------- 4. Класс нейронной сети QNet ----------------
class QNet(nn.Module):
    def __init__(self, input_dim=11, hidden=256, output_dim=3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)

# ---------------- 5. Класс агента QTrainer ----------------
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, *args):
        self.memory.append(tuple(args))
    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        return zip(*batch)
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self):
        self.model = QNet()
        self.target_model = QNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = ReplayMemory(BUFFER_SIZE)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = (self.epsilon - self.epsilon_min)/80
        self.steps = 0
        self.gamma = GAMMA
        self.update_target_every = 1000
        self.sync_target()
        self.loss_fn = nn.MSELoss()
    
    def sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state, train_mode=True):
        if train_mode and np.random.rand() < self.epsilon:
            move = random.randint(0,2)
            return move
        state0 = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(state0)
        move = torch.argmax(pred).item()
        return move
    
    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        # Q(s,a)
        q_vals = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        # Q'(s',a)
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0]
        target = rewards + self.gamma * max_next_q * (~dones)
        loss = self.loss_fn(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train_short_memory(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor([[action]], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.bool)
        
        q_val = self.model(state).gather(1, action).squeeze()
        with torch.no_grad():
            max_next_q = self.target_model(next_state).max(1)[0]
        target = reward + self.gamma * max_next_q * (~done)
        loss = self.loss_fn(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
    
    def save(self, path=MODEL_PATH):
        torch.save(self.model.state_dict(), path)
    def load(self, path=MODEL_PATH):
        self.model.load_state_dict(torch.load(path))
        self.sync_target()

# ---------------- 6. Функции обучения и запуска ----------------
def plot(scores, means):
    plt.figure("Snake DQN")
    plt.clf()
    plt.title('Результаты обучения')
    plt.xlabel('Игра')
    plt.ylabel('Счет')
    plt.plot(scores, label='Score')
    plt.plot(means, label='Mean(10)')
    plt.legend()
    plt.pause(0.001)

def human_action(event, cur_dir):
    # Человек управляет стрелками
    if event.key == pygame.K_LEFT:
        if cur_dir != Direction.RIGHT:
            return [0,0,1]
    if event.key == pygame.K_RIGHT:
        if cur_dir != Direction.LEFT:
            return [0,1,0]
    if event.key == pygame.K_UP:
        if cur_dir != Direction.DOWN:
            # Игнорируем UP как "прямо" если не вниз
            return [1,0,0] if cur_dir == Direction.UP else None
    if event.key == pygame.K_DOWN:
        if cur_dir != Direction.UP:
            return [1,0,0] if cur_dir == Direction.DOWN else None
    return None

# ---------------- 7. Основной цикл программы ----------------
def main():
    record = 0
    total_score = 0
    scores = []
    mean_scores = []
    game_idx = 0
    speed_idx = 0
    speed = SPEEDS[speed_idx]
    mode = 'train'    # 'train', 'demo', 'human'
    log = []
    
    agent = DQNAgent()
    game = SnakeGame()
    display = pygame.display.set_mode((game.w, game.h))
    pygame.display.set_caption("Змейка с ИИ (PyTorch DQN)")

    running = True
    show_plot = False
    plot_scores = []
    plot_means = []

    while running:
        game.reset()
        state_old = game.get_state()
        game_idx += 1
        score_this_game = 0
        done = False

        while not done:
            # -- Обработка событий Pygame (в том числе управление режимами)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    plt.close()
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    # Переключение режима
                    if event.key == pygame.K_SPACE:
                        mode = {'train':'demo','demo':'human','human':'train'}[mode]
                    # Сохранение/загрузка
                    if event.key == pygame.K_s:
                        agent.save()
                    if event.key == pygame.K_l:
                        agent.load()
                        agent.epsilon = agent.epsilon_min
                    # Скорость игры
                    if event.key == pygame.K_1: speed_idx = 0
                    if event.key == pygame.K_2: speed_idx = 1
                    if event.key == pygame.K_3: speed_idx = 2
                    if event.key == pygame.K_4: speed_idx = 3
                    speed = SPEEDS[speed_idx]
                    # Включение/выключение графика
                    if event.key == pygame.K_p:
                        show_plot = not show_plot

            # -- Выбор действия
            if mode == 'human':
                keys = pygame.key.get_pressed()
                action = [1,0,0] # По умолчанию "прямо"
                for event in pygame.event.get([pygame.KEYDOWN]):
                    user_action = human_action(event, game.direction)
                    if user_action is not None:
                        action = user_action
                        break
            else:
                action_idx = agent.act(state_old, train_mode=(mode=='train'))
                action = [0,0,0]
                action[action_idx] = 1

            # -- Игровой шаг
            reward, done, score = game.play_step(action)
            state_new = game.get_state()

            # -- Обучение
            if mode == 'train':
                agent.train_short_memory(state_old, np.argmax(action), reward, state_new, done)
                agent.remember(state_old, np.argmax(action), reward, state_new, done)
                if agent.steps % agent.update_target_every == 0:
                    agent.sync_target()
                agent.train_long_memory()
                agent.decay_epsilon()
                agent.steps += 1

            state_old = state_new
            score_this_game = score

            # -- Визуализация
            avg_score = np.mean(scores[-10:]) if scores else 0
            game.render(display, record, avg_score, speed, game_idx, mode)

            pygame.time.Clock().tick(speed)
        
        # -- После окончания игры
        scores.append(score_this_game)
        mean_scores.append(np.mean(scores[-10:]))
        record = max(record, score_this_game)
        total_score += score_this_game
        log.append(score_this_game)

        print(f"Игра {game_idx}, Счет: {score_this_game}, Рекорд: {record}, Epsilon: {agent.epsilon:.3f}, Режим: {mode}")

        if show_plot or game_idx % 5 == 0:
            plot(scores, mean_scores)
    
    pygame.quit()

if __name__ == '__main__':
    print('=== Snake DQN (PyTorch) ===')
    print('Управление:')
    print(' - 1/2/3/4: Скорость игры')
    print(' - SPACE: Переключить режим (train/demo/human)')
    print(' - S: Сохранить модель | L: Загрузить')
    print(' - P: Показать/скрыть график обучения')
    print(' - ESC/Quit: Выйти')
    main()
