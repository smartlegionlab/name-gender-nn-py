import sys

from model import NameNet

WEIGHTS_PATH = "weights.json"


def read_line(prompt="> "):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    raw = sys.stdin.buffer.readline()
    if not raw:
        return None
    return raw.decode("utf-8", errors="replace").strip()


def main():
    try:
        net = NameNet.load(WEIGHTS_PATH)
    except FileNotFoundError:
        print(f"File {WEIGHTS_PATH} not found. Run: python train.py")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}. Retrain the model: python train.py")
        sys.exit(1)

    if len(sys.argv) > 1:
        name = " ".join(sys.argv[1:])
        pred, prob = net.predict(name)
        print(f"{name} -> {pred}  (confidence {max(prob, 1 - prob):.1%})")
        return

    print("Enter a name (or 'exit' to quit):")
    while True:
        name = read_line("> ")
        if name is None:
            print()
            break
        if not name:
            continue
        if name.lower() in ("exit", "quit", "выход"):
            break
        pred, prob = net.predict(name)
        confidence = max(prob, 1 - prob)
        print(f"{name} -> {pred}  (confidence {confidence:.1%})")


if __name__ == "__main__":
    main()
