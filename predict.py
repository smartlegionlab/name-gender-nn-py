import argparse
import sys

from model import NameNet


def read_line(prompt="> "):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    raw = sys.stdin.buffer.readline()
    if not raw:
        return None
    return raw.decode("utf-8", errors="replace").strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Predict gender from a name")
    parser.add_argument("--weights", required=True, help="path to weights JSON")
    parser.add_argument("name", nargs="*", help="name to classify (optional)")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        net = NameNet.load(args.weights)
    except FileNotFoundError:
        print(f"File {args.weights} not found. Train it first.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.name:
        name = " ".join(args.name)
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
