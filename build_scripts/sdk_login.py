import sys

from layer import login_with_api_key


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        api_key = f.read().strip()
        login_with_api_key(api_key)
