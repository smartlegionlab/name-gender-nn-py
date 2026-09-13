import json
import math
import random

ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя_"
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
ALPHA_SIZE = len(ALPHABET)
MAX_LEN = 12
HIDDEN_SIZE = 40
INPUT_SIZE = 2 * MAX_LEN * ALPHA_SIZE + ALPHA_SIZE * ALPHA_SIZE


def sigmoid(z):
    if z < -500:
        return 0.0
    if z > 500:
        return 1.0
    return 1 / (1 + math.exp(-z))


def suffix_idx(name):
    s = name.lower()
    a = CHAR_TO_IDX.get(s[-2], 0) if len(s) >= 2 else 0
    b = CHAR_TO_IDX.get(s[-1], 0) if len(s) >= 1 else 0
    return a * ALPHA_SIZE + b


def encode_sparse(name):
    name = name.lower()
    n = len(name)
    out = []

    for pos, ch in enumerate(name[:MAX_LEN]):
        idx = CHAR_TO_IDX.get(ch, CHAR_TO_IDX["_"])
        out.append((pos * ALPHA_SIZE + idx, 1.0))

    for k in range(1, MAX_LEN + 1):
        ch = name[-k] if k <= n else "_"
        idx = CHAR_TO_IDX.get(ch, CHAR_TO_IDX["_"])
        offset = MAX_LEN * ALPHA_SIZE + (k - 1) * ALPHA_SIZE
        out.append((offset + idx, 1.0))

    suffix_offset = 2 * MAX_LEN * ALPHA_SIZE
    out.append((suffix_offset + suffix_idx(name), 1.0))

    return out


class NameNet:
    def __init__(self, seed=42):
        random.seed(seed)
        self.W1 = [
            [random.uniform(-1, 1) / math.sqrt(INPUT_SIZE) for _ in range(INPUT_SIZE)]
            for _ in range(HIDDEN_SIZE)
        ]
        self.B1 = [0.0] * HIDDEN_SIZE
        self.W2 = [
            random.uniform(-1, 1) / math.sqrt(HIDDEN_SIZE)
            for _ in range(HIDDEN_SIZE)
        ]
        self.B2 = 0.0

    def forward(self, x_sparse):
        h = [0.0] * HIDDEN_SIZE
        for i in range(HIDDEN_SIZE):
            Wi = self.W1[i]
            z = self.B1[i]
            for j, v in x_sparse:
                z += Wi[j] * v
            h[i] = sigmoid(z)

        z_out = self.B2
        for i in range(HIDDEN_SIZE):
            z_out += self.W2[i] * h[i]
        y = sigmoid(z_out)

        return h, y

    def train_step(self, x_sparse, target, lr):
        h, y = self.forward(x_sparse)
        error = target - y
        d_y = error * y * (1 - y)

        for i in range(HIDDEN_SIZE):
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
        x_sparse = encode_sparse(name)
        _, y = self.forward(x_sparse)
        label = "female" if y > 0.5 else "male"
        return label, y

    def save(self, path):
        data = {
            "alphabet": ALPHABET,
            "max_len": MAX_LEN,
            "hidden_size": HIDDEN_SIZE,
            "input_size": INPUT_SIZE,
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

        if (
            data["alphabet"] != ALPHABET
            or data["max_len"] != MAX_LEN
            or data["hidden_size"] != HIDDEN_SIZE
        ):
            raise ValueError("weights.json is incompatible with current model")

        net = cls.__new__(cls)
        net.W1 = data["W1"]
        net.B1 = data["B1"]
        net.W2 = data["W2"]
        net.B2 = data["B2"]
        return net
