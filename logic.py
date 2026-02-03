import os

def next_directory(directory, prefix):
    i = 0
    if os.path.exists(f"{directory}/{prefix}"):
        i += 1

    while os.path.exists(f"{directory}/{prefix}_{i}"):
        i += 1

    if i == 0:
        new_dir = prefix
        return new_dir
    else:
        new_dir = f"{prefix}_{i}"
        return new_dir


if __name__ == "__main__":
    print(next_directory("training/trainer_16", "base_model"))
