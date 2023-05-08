from model import Model
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
CUDA_LAUNCH_BLOCKING=1


class Trainer:
    def __init__(self):
        self.args = Args().graph()

        # seed = torch.seed()
        seed = 258116326138800
        print(f'torch_seed:{seed}')

        torch.manual_seed(seed)
        self.model = Model(in_features=self.args.in_feature, out_features=64).to(self.args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.best_acc = 0

    def train(self, x, a, y):
        for epoch in range(self.args.epochs):
            self.model.train()

            x = torch.Tensor(x).to(self.args.device)
            a = torch.Tensor(a).to(self.args.device)
            y = torch.Tensor(y).to(self.args.device)

            self.optimizer.zero_grad()
            yhat = self.model(x, a)

            # loss = self.criterion(yhat, y)

            loss = torch.tensor(0., dtype=torch.float32, device=self.args.device)

            for i in range(yhat.shape[1]):
                loss += self.criterion(yhat[:,i,:], y[:, i])
                print(i)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            test_acc = self.metric(yhat, y)
            if test_acc>self.best_acc:
                self.best_acc = test_acc
            print(f'Epoch: {epoch} | val_acc: {test_acc} | best_val_acc: {self.best_acc}')

    def metric(self, output, target):
        acc = 0
        out = torch.argmax(output, dim=2)

        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if out[i,j] == target[i, j]:
                    acc += 1
        acc = acc/(out.shape[0]*out.shape[1])*100
        return acc


class Args:
    def graph(self):
        parser = argparse.ArgumentParser(add_help=False)

        parser.add_argument('--in_feature', default=9, type=int)
        parser.add_argument('--out_feature', default=8, type=int)

        parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                            type=str)
        parser.add_argument('--lr', default=0.0005, type=float)
        parser.add_argument('--epochs', default=100, type=int)

        args = parser.parse_args()
        return args

