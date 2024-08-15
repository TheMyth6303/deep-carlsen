import torch
import torch.nn as nn
import torch.nn.functional as f
import os


MODEL_HYPERPARAMS = {
    'global_nodes': 32,
    'piece_nodes_1': 64,
    'piece_nodes_2': 32,
    'attack_defend_nodes': 32,
    'overall': 32,
    'loss': nn.MSELoss()
}


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.global_layer = nn.Linear(15, MODEL_HYPERPARAMS['global_nodes'])
        self.piece_layer_1 = nn.Linear(190, MODEL_HYPERPARAMS['piece_nodes_1'])
        self.piece_layer_2 = nn.Linear(MODEL_HYPERPARAMS['piece_nodes_1'], MODEL_HYPERPARAMS['piece_nodes_2'])
        self.attack_defend_layer = nn.Linear(128, MODEL_HYPERPARAMS['attack_defend_nodes'])
        self.overall = nn.Linear(MODEL_HYPERPARAMS['global_nodes'] +
                                 MODEL_HYPERPARAMS['piece_nodes_2'] +
                                 MODEL_HYPERPARAMS['attack_defend_nodes'],
                                 MODEL_HYPERPARAMS['overall'])
        self.output = nn.Linear(MODEL_HYPERPARAMS['overall'], 1)
        self._initialize_params()

    def _initialize_params(self):
        for layer in self.modules():
            if layer != self:
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.global_layer.weight[0][5], 9)
        nn.init.constant_(self.global_layer.weight[0][10], -9)
        nn.init.constant_(self.global_layer.weight[0][6], 5)
        nn.init.constant_(self.global_layer.weight[0][11], -5)
        nn.init.constant_(self.global_layer.weight[0][7], 3)
        nn.init.constant_(self.global_layer.weight[0][12], -3)
        nn.init.constant_(self.global_layer.weight[0][8], 3)
        nn.init.constant_(self.global_layer.weight[0][13], -3)
        nn.init.constant_(self.global_layer.weight[0][9], 1)
        nn.init.constant_(self.global_layer.weight[0][14], -1)
        nn.init.constant_(self.global_layer.bias[0], 80)
        nn.init.constant_(self.overall.weight[0][0], 1)
        nn.init.constant_(self.output.weight[0][0], 1)
        nn.init.constant_(self.output.bias, -80)

    def forward(self, x):
        global_features = x[:15]
        global_features = self.global_layer(global_features)
        global_features = f.relu(global_features)

        piece_features = x[15:205]
        piece_features = self.piece_layer_1(piece_features)
        piece_features = f.relu(piece_features)
        piece_features = self.piece_layer_2(piece_features)
        piece_features = f.relu(piece_features)

        attack_defend_features = x[205:]
        attack_defend_features = self.attack_defend_layer(attack_defend_features)
        attack_defend_features = f.relu(attack_defend_features)

        x = torch.concat((global_features, piece_features, attack_defend_features))
        x = self.overall(x)
        x = f.relu(x)
        x = self.output(x)
        x = f.tanh(x/10)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


def testing():
    pass


if __name__ == '__main__':
    testing()
