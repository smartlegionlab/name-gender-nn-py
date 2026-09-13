import argparse
import json
import time

from model import NameNet


LR = 0.1
EPOCHS = 8000


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = []
    for name in raw["female"]:
        data.append((name, 1))
    for name in raw["male"]:
        data.append((name, 0))

    return raw["alphabet"], data


def parse_args():
    parser = argparse.ArgumentParser(description="Train name-gender classifier")
    parser.add_argument("--data", required=True, help="path to dataset JSON")
    parser.add_argument("--out", required=True, help="path to save weights JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    alphabet, data = load_data(args.data)
    print(f"Loaded {len(data)} names from {args.data}")
    print(f"Alphabet size: {len(alphabet)}")

    net = NameNet(alphabet=alphabet, seed=42)
    encoded = [(net.encode_sparse(n), t) for n, t in data]

    t0 = time.time()
    for epoch in range(EPOCHS):
        total_error = 0.0
        for x_sparse, target in encoded:
            total_error += net.train_step(x_sparse, target, LR)

        if epoch % 500 == 0:
            elapsed = time.time() - t0
            print(f"epoch {epoch:5}  error={total_error:.6f}  t={elapsed:.1f}s")

    correct = sum(
        1 for name, target in data
        if (net.predict(name)[1] > 0.5) == (target == 1)
    )
    print(f"\nTrain accuracy: {correct}/{len(data)} = {100 * correct / len(data):.1f}%")

    net.save(args.out)
    print(f"Weights saved to {args.out}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
