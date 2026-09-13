import json
import math
import random

MAX_LEN = 12
HIDDEN_SIZE = 40


def sigmoid(z):
    if z < -500:
        return 0.0
    if z > 500:
        return 1.0
    return 1 / (1 + math.exp(-z))


class NameNet:
    def __init__(self, alphabet, seed=42):
        self.alphabet = alphabet
        self.char_to_idx = {c: i for i, c in enumerate(alphabet)}
        self.alpha_size = len(alphabet)
        self.max_len = MAX_LEN
        self.hidden_size = HIDDEN_SIZE
        self.input_size = 2 * MAX_LEN * self.alpha_size + self.alpha_size ** 2

        random.seed(seed)
        self.W1 = [
            [random.uniform(-1, 1) / math.sqrt(self.input_size)
             for _ in range(self.input_size)]
            for _ in range(self.hidden_size)
        ]
        self.B1 = [0.0] * self.hidden_size
        self.W2 = [
            random.uniform(-1, 1) / math.sqrt(self.hidden_size)
            for _ in range(self.hidden_size)
        ]
        self.B2 = 0.0

    def suffix_idx(self, name):
        s = name.lower()
        a = self.char_to_idx.get(s[-2], 0) if len(s) >= 2 else 0
        b = self.char_to_idx.get(s[-1], 0) if len(s) >= 1 else 0
        return a * self.alpha_size + b

    def encode_sparse(self, name):
        name = name.lower()
        n = len(name)
        out = []
        pad = self.char_to_idx["_"]

        for pos, ch in enumerate(name[:self.max_len]):
            idx = self.char_to_idx.get(ch, pad)
            out.append((pos * self.alpha_size + idx, 1.0))

        for k in range(1, self.max_len + 1):
            ch = name[-k] if k <= n else "_"
            idx = self.char_to_idx.get(ch, pad)
            offset = self.max_len * self.alpha_size + (k - 1) * self.alpha_size
            out.append((offset + idx, 1.0))

        suffix_offset = 2 * self.max_len * self.alpha_size
        out.append((suffix_offset + self.suffix_idx(name), 1.0))
        return out

    def forward(self, x_sparse):
        h = [0.0] * self.hidden_size
        for i in range(self.hidden_size):
            Wi = self.W1[i]
            z = self.B1[i]
            for j, v in x_sparse:
                z += Wi[j] * v
            h[i] = sigmoid(z)

        z_out = self.B2
        for i in range(self.hidden_size):
            z_out += self.W2[i] * h[i]
        y = sigmoid(z_out)
        return h, y

    def train_step(self, x_sparse, target, lr):
        h, y = self.forward(x_sparse)
        error = target - y
        d_y = error * y * (1 - y)

        for i in range(self.hidden_size):
            hi = h[i]
            d_hi = d_y * self.W2[i] * hi * (1 - hi)
            Wi = self.W1[i]
            for j, v in x_sparse:
                Wi[j] += lr * d_hi * v
            self.B1[i] += lr * d_hi
            self.W2[i] += lr * d_y * hi

        self.B2 += lr * d_y
        return error * error

    def predict(self, name):
        x_sparse = self.encode_sparse(name)
        _, y = self.forward(x_sparse)
        label = "female" if y > 0.5 else "male"
        return label, y

    def save(self, path):
        data = {
            "alphabet": self.alphabet,
            "max_len": self.max_len,
            "hidden_size": self.hidden_size,
            "input_size": self.input_size,
            "W1": self.W1,
            "B1": self.B1,
            "W2": self.W2,
            "B2": self.B2,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        alphabet = data["alphabet"]
        if data["max_len"] != MAX_LEN or data["hidden_size"] != HIDDEN_SIZE:
            raise ValueError("weights file is incompatible with current model")

        net = cls.__new__(cls)
        net.alphabet = alphabet
        net.char_to_idx = {c: i for i, c in enumerate(alphabet)}
        net.alpha_size = len(alphabet)
        net.max_len = data["max_len"]
        net.hidden_size = data["hidden_size"]
        net.input_size = data["input_size"]
        net.W1 = data["W1"]
        net.B1 = data["B1"]
        net.W2 = data["W2"]
        net.B2 = data["B2"]
        return net
