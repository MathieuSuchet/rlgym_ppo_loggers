import json

from setuptools import setup

if __name__ == "__main__":
    with open("setup.json", "r") as f:
        setup(
            **json.load(f)
        )