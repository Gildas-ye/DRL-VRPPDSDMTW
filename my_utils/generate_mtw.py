import random


def generate_time_windows(vrp_size):
    time_windows = []

    for _ in range(vrp_size):
        num_windows = random.randint(1, 5)
        print(f'num windows = {num_windows}')
        windows = []
        previous_lt = 420 - 60
        for _ in range(num_windows):
            if previous_lt + 60 > 1200-60:
                continue
            et = random.randint(previous_lt + 60, (previous_lt + 60)+ ((1200 - 60 - (previous_lt + 60))//2))
            lt = random.randint(et + 60, 1200)
            windows.append([et, lt])
            previous_lt = lt
        time_windows.append(windows)

    return time_windows
