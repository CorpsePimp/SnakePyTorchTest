Оцени данный код от 1 до 10
# Импорты библиотек
import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Константы и настройки
# Размеры игрового поля
WIDTH = 20
HEIGHT = 20
CELL_SIZE = 20
# Цвета
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
# Параметры обучения
LR = 0.001
GAMMA = 0.9
EPSILON_DECAY = 0.95
MEMORY_SIZE = 100_000
BATCH_SIZE = 1000
UPDATE_TARGET_EVERY = 1000

# Класс игры "Змейка"
class SnakeGame:
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Сброс игры в начальное состояние"""
        self.snake = [(WIDTH//2, HEIGHT//2)]
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.game_over = False
        
    def _place_food(self):
        """Размещение еды на поле"""
        while True:
            pos = (random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1))
            if pos not in self.snake:
                return pos
            
    def get_state(self):
        """Получение состояния для нейросети (11 признаков)"""
        head = self.snake[0]
        dir_l = self.direction
        dir_r = (-dir_l[0], -dir_l[1]) if dir_l[0] else (dir_l[1], dir_l[0])
        dir_u = (dir_l[1], dir_l[0]) if dir_l[1] else (dir_l[0], dir_l[1])
        
        # Опасности в 3 направлениях
        danger_straight = self._is_collision((head[0] + dir_l[0], head[1] + dir_l[1]))
        danger_right = self._is_collision((head[0] + dir_r[0], head[1] + dir_r[1]))
        danger_left = self._is_collision((head[0] + dir_u[0], head[1] + dir_u[1]))
        
        # Положение еды относительно змейки
        food_dir = np.array(self.food) - np.array(head)
        food_up = food_dir[1] < 0
        food_down = food_dir[1] > 0
        food_left = food_dir[0] < 0
        food_right = food_dir[0] > 0
        
        return np.array([
            danger_straight,
            danger_right,
            danger_left,
            dir_l == (0, -1),  # Движение вверх
            dir_l == (0, 1),   # Движение вниз
            dir_l == (-1, 0),  # Движение влево
            dir_l == (1, 0),   # Движение вправо
            food_up,
            food_down,
            food_left,
            food_right
        ], dtype=np.float32)
    
    def _is_collision(self, pos):
        """Проверка столкновения с границами или телом"""
        x, y = pos
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return True
        return pos in self.snake
    
    def step(self, action):
        """Выполнение шага игры"""
        self.steps += 1
        reward = 0
        
        # Поворот направления
        if action == 1:  # Направо
            self.direction = (self.direction[1], -self.direction[0])
        elif action == 2: # Налево
            self.direction = (-self.direction[1], self.direction[0])
        
        # Движение головы
        new_head = (self.snake[0][0] + self.direction[0], 
                   self.snake[0][1] + self.direction[1])
        
        # Проверка столкновений
        if self._is_collision(new_head) or self.steps > 100*len(self.snake):
            self.game_over = True
            reward = -10
            return reward, self.game_over
        
        # Добавление новой головы
        self.snake.insert(0, new_head)
        
        # Проверка поедания еды
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()
            reward = -0.1  # Штраф за шаг
            
        # Награда за движение к еде
        prev_dist = abs(self.food[0] - self.snake[0][0]) + abs(self.food[1] - self.snake[0][1])
        new_dist = abs(self.food[0] - new_head[0]) + abs(self.food[1] - new_head[1])
        reward += 1 if new_dist < prev_dist else -1
        
        return reward, self.game_over

# Класс нейронной сети
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Класс агента
class DQNAgent:
    def __init__(self):
        self.model = DQN(11, 256, 3)
        self.target_model = DQN(11, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0
        self.losses = []
        
    def get_action(self, state):
        """Выбор действия с учетом ε-жадной стратегии"""
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def update_epsilon(self, episode):
        """Обновление значения epsilon"""
        self.epsilon = max(0.01, 0.1 + (1.0 - 0.1) * np.exp(-0.001 * episode))
        
    def remember(self, state, action, reward, next_state, done):
        """Сохранение опыта в память"""
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self):
        """Обучение на случайной выборке из памяти"""
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * GAMMA * next_q
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

# Функция обучения
def train():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH*CELL_SIZE, HEIGHT*CELL_SIZE))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 20)
    
    game = SnakeGame()
    agent = DQNAgent()
    total_rewards = []
    avg_scores = []
    best_score = 0
    total_steps = 0
    
    # Основной цикл обучения
    for episode in count(1):
        game.reset()
        total_reward = 0
        while not game.game_over:
            # Получение состояния и действия
            state = game.get_state()
            action = agent.get_action(state)
            
            # Выполнение действия
            reward, done = game.step(action)
            next_state = game.get_state() if not done else None
            
            # Сохранение опыта
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            
            # Обучение
            agent.replay()
            
            # Обновление целевой сети
            if total_steps % UPDATE_TARGET_EVERY == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())
            
            total_steps += 1
            
            # Отрисовка
            screen.fill(BLACK)
            for idx, (x, y) in enumerate(game.snake):
                color = GREEN if idx == 0 else (0, 200, 0)
                pygame.draw.rect(screen, color, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE-1, CELL_SIZE-1))
            pygame.draw.rect(screen, RED, (game.food[0]*CELL_SIZE, game.food[1]*CELL_SIZE, CELL_SIZE-1, CELL_SIZE-1))
            
            # Отображение информации
            text = font.render(f'Score: {game.score} Best: {best_score} Eps: {agent.epsilon:.2f}', True, WHITE)
            screen.blit(text, (10, 10))
            pygame.display.flip()
            clock.tick(60)
            
        # Обновление статистики
        total_rewards.append(total_reward)
        avg_score = np.mean(total_rewards[-10:])
        avg_scores.append(avg_score)
        best_score = max(best_score, game.score)
        
        # Обновление epsilon
        agent.update_epsilon(episode)
        
        # Логирование
        print(f'Episode: {episode}, Score: {game.score}, Avg: {avg_score:.2f}, Eps: {agent.epsilon:.2f}')
        
        # Построение графика
        if episode % 10 == 0:
            plt.clf()
            plt.plot(avg_scores)
            plt.title('Средний счет за последние 10 игр')
            plt.xlabel('Игры')
            plt.ylabel('Счет')
            plt.pause(0.01)
            
    pygame.quit()

if __name__ == '__main__':
    train()