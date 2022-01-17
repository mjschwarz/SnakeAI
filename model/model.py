import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    """
    Represent a linear (deep-learning) QNet
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the linear QNet
        :param input_size: <int> size of input layer
        :param hidden_size: <int> size of hidden layer
        :param output_size: <int> size of output layer
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Feed forward
        :param x: tensor
        :return:
        """
        x = F.relu(self.linear1(x)) # has activation function
        x = self.linear2(x) # no activation function
        return x

    def save(self, file_name='model.pth'):
        """
        Save the model
        :param file_name: <str> file to save model to
        :return: None
        """
        model_folder_path = './model'
        # create directory
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        """
        Load the previously saved model
        :param file_name: <str> file to load model from
        :return: True if file exists, False if not
        """
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        # use save file
        if os.path.isfile(file_name):
            self.load_state_dict(torch.load(file_name))
            self.eval()
            print('Loading existing model state')
            return True
        # start from scratch
        print('No existing model state found...starting from scratch')
        return False

class QTrainer:
    """
    Represent neural net trainer
    """
    def __init__(self, model, lr, gamma):
        """
        Initialize the trainer
        :param model: model being trained
        :param lr: <float> learning rate
        :param gamma: <float> discount rate
        """
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        """
        Take a training step
        Bellman equation: Q_new = R + y * max(next predicted Q value)
        :param state: <np.array> current danger, movement, and food states
        :param action: <array> direction to move [straight, left, right]
        :param reward: <int> reward amount
        :param next_state: <np.array> next danger, movement, and food states
        :param game_over: <bool> whether game is finished
        :return: None
        """
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        # reshape as (1, x) if not (n, x) already
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = (game_over, )
        # predicted Q values using current state
        pred = self.model(state)
        # new Q values using Bellman equation
        target = pred.clone()
        for index in range(len(game_over)):
            Q_new = reward[index]
            if not game_over[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
            target[index][torch.argmax(action).item()] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()