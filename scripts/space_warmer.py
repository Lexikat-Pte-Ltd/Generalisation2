import time
import os
import random


def create_file(file_name, size):
    print(f"Creating... {file_name}")
    with open(file_name, "wb") as f:
        f.write(os.urandom(size))


def delete_random_file():
    files = [f for f in os.listdir(".") if os.path.isfile(f) and f.startswith("data_")]
    file = random.choice(files)

    print(f"Deleting... {file}")

    if files:
        os.remove(file)


def main():
    print("Starting...")

    total_files = 0
    counter = 0

    if counter == 9:
        for i in range(10):
            file_name = f"data_{int(time.time()*10)}.bin"
            create_file(file_name, 102400)  # 0.1 MB = 100 KB = 102400 bytes
            total_files += 1
            time.sleep(0.05)

    if counter == 19:
        for i in range(10):
            delete_random_file()
            total_files -= 1

    while True:
        perform_creation = random.choice(
            [True, False]
            + [True]
            * (
                100 - total_files
            )  # Increase the probability of not creating files if it theres already a lot of them
        )

        if perform_creation:
            file_name = f"data_{int(time.time()*10)}.bin"
            create_file(file_name, 102400)  # 0.1 MB = 100 KB = 102400 bytes
            total_files += 1
            time.sleep(0.1)

        perform_deletion = random.choice(
            [True, False]
            + [True]
            * (
                total_files - 100 if total_files - 100 > 0 else 0
            )  # Increase the probability of deleting files if it theres already a lot of them
        )

        if perform_deletion:  # Randomly decide whether to delete a file
            delete_random_file()
            total_files -= 1

        counter += 1

        if counter > 20:
            print("Total files ", total_files)
            counter = 0


if __name__ == "__main__":
    main()
