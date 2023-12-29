import dill


def save(obj, filename: str):
    with open(filename, 'wb') as file:
        dill.dump(obj, file)


def load(filename: str):
    with open(filename, 'rb') as file:
        return dill.load(file)
