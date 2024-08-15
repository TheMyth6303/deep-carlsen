from chess import *
from model import *
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
# from plot import plot


AGENT_HYPERPARAMS = {
    'depth': 3,
    'lr': 0.5,
    'lambda': 0.7,
    'gamma': 0.99,
    'loss': nn.MSELoss()
}


class MoveOrderer:
    def __init__(self, model):
        self.model = model

    def _captures_first(self, moves, board):
        pass

    def _SEE(self, moves, board):
        see_scores = {}

        for move in moves:
            if board.cells[move.cell.row][move.cell.col] is not None:
                temp = board.apply_move(move)[0]
                see_scores[move] = self.model(temp.get_state(temp.side_to_move)).item() * (1 - 2*board.side_to_move)

        ordered_moves = sorted(moves, key=lambda x: see_scores.get(x, 0))
        if board.side_to_move == 0:
            ordered_moves = reversed(ordered_moves)

        return ordered_moves

    def order(self, moves, board):
        return self._SEE(moves, board)


class Agent:
    def __init__(self):
        self.model = DNN()
        self.episodes = 0
        self.epsilon = 0
        self.gamma = AGENT_HYPERPARAMS['gamma']
        self.lamda = AGENT_HYPERPARAMS['lambda']
        self.loss = AGENT_HYPERPARAMS['loss']
        self.optimizer = optim.AdamW(self.model.parameters(), lr=AGENT_HYPERPARAMS['lr'])
        self.move_orderer = MoveOrderer(self.model)

    def _q_search(self, board):
        # todo
        pass

    def _minimax_search_alpha_beta(self, board, depth, alpha, beta, searched):
        # todo: add move ordering
        side = board.side_to_move
        if depth == 0:
            searched[0] += 1
            with torch.no_grad():
                return None, self.model(board.get_state(side)).item() * (1 - 2*side)

        if side == 0:
            # white to move, wants to maximize evaluation.
            best_score = -2.0
            best_move = None
            moves = board.legal_moves(side)
            # moves = self.move_orderer.order(moves, board)
            for move in moves:
                next_board, reward, done = board.apply_move(move)
                if done:
                    if reward == 1:
                        return move, 1.0
                    else:
                        score = 0.0
                else:
                    score = self._minimax_search_alpha_beta(next_board, depth-1, alpha, beta, searched)[1]
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return best_move, best_score

        else:
            # black to move, wants to minimize evaluation.
            best_score = 2.0
            best_move = None
            moves = board.legal_moves(side)
            # moves = self.move_orderer.order(moves, board)
            for move in moves:
                next_board, reward, done = board.apply_move(move)
                if done:
                    if reward == 1:
                        return move, -1.0
                    else:
                        score = 0.0
                else:
                    score = self._minimax_search_alpha_beta(next_board, depth-1, alpha, beta, searched)[1]
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return best_move, best_score

    def get_action(self, board, greedy=True):
        if greedy and random.random() < self.epsilon:
            move = random.choice(board.legal_moves(board.side_to_move))
            return move, None
        else:
            searched = [0]
            best_move, evaluation = self._minimax_search_alpha_beta(board, AGENT_HYPERPARAMS['depth'], -2.0, 2.0, searched)
            print(searched[0])
            return best_move, evaluation

    def train(self, states, rewards, next_states, eligibilities, done, losses):
        with torch.no_grad():
            if done:
                td_pred = rewards[-1] - self.model(states[-1])
            else:
                td_pred = rewards[-1] + self.gamma * self.model(next_states[-1]) - self.model(states[-1])

        if states[-1] in eligibilities:
            eligibilities[states[-1]] += 1
        else:
            eligibilities[states[-1]] = 1

        for idx in range(len(states)):
            pred = self.model(states[idx])
            self.optimizer.zero_grad()
            loss = self.loss(td_pred, pred)
            if idx == len(states) - 1:
                losses.append(loss.detach().item())
            loss.backward()
            self.optimizer.step()
            eligibilities[states[-1]] = self.gamma*self.lamda*eligibilities[states[-1]]

    def analyze_performance(self):
        # todo: figure out
        pass


def train():

    agent = Agent()
    losses = deque(maxlen=100)

    agent.model.load_state_dict(torch.load("models/model.pth"))

    while True:

        if agent.episodes % 100 == 0:
            agent.analyze_performance()

        board = Board()
        agent.episodes += 1

        states = []
        rewards = []
        next_states = []
        eligibilities = {}

        while True:

            if agent.episodes % 10 == 1:
                board.display()

            states.append(board.get_state(board.side_to_move))
            action, score = agent.get_action(board)
            board, reward, done = board.apply_move(action)
            next_states.append(board.get_state(1 - board.side_to_move))
            rewards.append(reward)

            agent.train(states, rewards, next_states, eligibilities, done, losses)

            if done:
                if reward == 1:
                    print('won!')
                print(agent.episodes)
                agent.model.save()
                # plot(losses, agent.episodes)
                agent.episodes += 1
                # agent.epsilon = 1 / int(1 + agent.episodes/200)
                agent.epsilon = 1 / agent.episodes
                break


if __name__ == '__main__':
    train()
    # testing()
