import argparse
import json
import re
import sys


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_regex(alphabet):
    chars = alphabet.replace("_", "")
    return re.compile(f"^[{re.escape(chars)}]+$")


def check_alphabet(data, pattern):
    bad = []
    for gender in ("female", "male"):
        for name in data[gender]:
            if not pattern.match(name):
                bad.append((gender, name))
    return bad


def check_duplicates(data):
    duplicates = {}
    for gender in ("female", "male"):
        names = data[gender]
        seen = set()
        repeated = []
        for n in names:
            if n in seen and n not in repeated:
                repeated.append(n)
            seen.add(n)
        duplicates[gender] = repeated
    return duplicates


def check_cross_duplicates(data):
    female = set(data["female"])
    male = set(data["male"])
    return sorted(female & male)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a names dataset")
    parser.add_argument("--data", required=True, help="path to dataset JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load(args.data)
    alphabet = data["alphabet"]
    pattern = build_regex(alphabet)

    print(f"File: {args.data}")
    print(f"Alphabet size: {len(alphabet)}")
    print(f"Loaded {len(data['female'])} female and {len(data['male'])} male names")
    print()

    bad = check_alphabet(data, pattern)
    if bad:
        print("Names with characters outside the alphabet:")
        for gender, name in bad:
            print(f"  [{gender}] {name!r}")
    else:
        print("Alphabet check: OK")

    duplicates = check_duplicates(data)
    any_duplicates = False
    for gender, repeated in duplicates.items():
        if repeated:
            any_duplicates = True
            print(f"Duplicates in {gender}:")
            for name in repeated:
                print(f"  {name}")
    if not any_duplicates:
        print("Duplicate check: OK")

    cross = check_cross_duplicates(data)
    if cross:
        print("Names present in both female and male:")
        for name in cross:
            print(f"  {name}")
    else:
        print("Cross-gender check: OK")

    if bad or any_duplicates or cross:
        sys.exit(1)

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
