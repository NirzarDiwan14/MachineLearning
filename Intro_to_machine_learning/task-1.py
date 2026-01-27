import random
data = [
    {"age": 25, "salary": 50000, "label": 0},
    {"age": 30, "salary": 60000, "label": 1},
    {"age": 22, "salary": 45000, "label": 0},
    {"age": 28, "salary": 52000, "label": 1},
    {"age": 35, "salary": 70000, "label": 1},
    {"age": 40, "salary": 80000, "label": 1},
    {"age": 23, "salary": 48000, "label": 0},
    {"age": 31, "salary": 62000, "label": 1},
    {"age": 27, "salary": 51000, "label": 0},
    {"age": 45, "salary": 90000, "label": 1}
]


def manual_train_test_split_list(data, test_size=0.2, seed=None):
    if seed is not None:
        random.seed(seed)

    data = data.copy()
    random.shuffle(data)

    split_index = int(len(data) * (1 - test_size))

    train_data = data[:split_index]
    test_data  = data[split_index:]

    return train_data, test_data


training_data,testing_data = manual_train_test_split_list(data)
print(training_data)
print("Testing data: ")
print(testing_data)
