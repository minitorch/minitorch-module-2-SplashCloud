"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""
import random

import minitorch


class Network(minitorch.Module):
    def __init__(self, hidden_layers, in_size, out_size, hidden_size=8):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.layers = []
        hidden_in_size = in_size
        hidden_out_size = in_size
        for i in range(1, hidden_layers+1):
            hidden_out_size = hidden_size
            self.layers.append(
                self.add_module(f"layer{i}", Linear(hidden_in_size, hidden_out_size))
            )
            hidden_in_size = hidden_size
        self.output = Linear(hidden_out_size, out_size)
        

    def forward(self, x):
        middle = x
        for i in range(self.hidden_layers):
            middle = [h.relu() for h in self.layers[i].forward(middle)]
        return self.output.forward(middle)[0].sigmoid()
        # middle = [h.relu() for h in self.layer1.forward(x)]
        # end = [h.relu() for h in self.layer2.forward(middle)]
        # return self.layer3.forward(end)[0].sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = [] # in_size x out_size
        self.bias = [] # 1 x out_size
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    def forward(self, inputs):
        # input: in_size
        assert len(inputs) == len(self.weights)
        in_size = len(self.weights)
        out_size = len(self.weights[0])
        res = [] # m x out_size
        # inputs X weights
        for j in range(out_size):
            ele = 0.0
            for k in range(in_size):
                ele += inputs[k] * self.weights[k][j].value
            res.append(ele)
        # + bias
        for j in range(out_size):
            res[j] += self.bias[j].value
        return res



def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class ScalarTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers, in_size=2, out_size=1)

    def run_one(self, x):
        return self.model.forward(
            (minitorch.Scalar(x[0], name="x_1"), minitorch.Scalar(x[1], name="x_2"))
        )

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers, in_size=2, out_size=1)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            loss = 0
            for i in range(data.N):
                x_1, x_2 = data.X[i]
                y = data.y[i]
                x_1 = minitorch.Scalar(x_1)
                x_2 = minitorch.Scalar(x_2)
                out = self.model.forward((x_1, x_2))

                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0
                else:
                    prob = -out + 1.0
                    correct += 1 if out.data < 0.5 else 0
                loss = -prob.log()
                (loss / data.N).backward()
                total_loss += loss.data

            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    ScalarTrain(HIDDEN).train(data, RATE)
