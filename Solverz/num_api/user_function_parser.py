import os
from pathlib import Path


def validate_module_paths(paths):
    """
    Validate that each path in the provided list points to a valid Python module.

    :param paths: List of paths
    :return: List of validated paths
    :raises ValueError: If a path is invalid or not a Python module
    """
    valid_paths = []
    for path in paths:
        # Check if the path exists
        if not os.path.exists(path):
            raise ValueError(f"The path {path} does not exist.")

            # Check if the path is a file
            if not os.path.isfile(path):
                raise ValueError(f"The path {path} is not a file.")

            # Check if the file is a Python file
            if not path.endswith('.py'):
                raise ValueError(f"The file {path} is not a Python file.")

        # If all checks pass, add the path to the valid paths list
        valid_paths.append(path)

    return valid_paths


def add_my_module(paths, filename='user_modules.txt'):
    """
        Save user-provided module paths to a specified file, but validate the paths before saving.
        If a path already exists in the file, it will not be added again.

        :param paths: List of user-provided module paths
        :param filename: Name of the file to save, default is 'user_modules.txt'
        """
    try:
        # Validate paths
        validated_paths = validate_module_paths(paths)
    except ValueError as e:
        print(e)
        return

    # Get the path to the .Solverz directory in the user's home directory
    user_home = str(Path.home())
    solverz_dir = os.path.join(user_home, '.Solverz')

    # Create the .Solverz directory if it does not exist
    if not os.path.exists(solverz_dir):
        os.makedirs(solverz_dir)

    # Define the full path to the file
    file_path = os.path.join(solverz_dir, filename)

    # Read existing paths from the file
    existing_paths = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            existing_paths = set(line.strip() for line in file)

    # Write the new paths to the file, but only if they are not already present
    with open(file_path, 'a') as file:
        for path in validated_paths:
            if path not in existing_paths:
                file.write(f'{path}\n')
                existing_paths.add(path)


def load_my_module_paths(filename='user_modules.txt'):
    """
    Load module paths from a specified file in the .Solverz directory in the user's home directory.

    :param filename: Name of the file to load, default is 'user_modules.txt'
    :return: List of module paths
    """
    user_home = str(Path.home())
    solverz_dir = os.path.join(user_home, '.Solverz')
    file_path = os.path.join(solverz_dir, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        return []

    # Read and return the list of paths
    with open(file_path, 'r') as file:
        paths = [line.strip() for line in file]

    return paths


def reset_my_module_paths(filename='user_modules.txt'):
    """
    Reset the user_modules.txt file by clearing its content.

    :param filename: Name of the file to reset, default is 'user_modules.txt'
    """
    # Get the path to the .Solverz directory in the user's home directory
    user_home = str(Path.home())
    solverz_dir = os.path.join(user_home, '.Solverz')

    # Define the full path to the file
    file_path = os.path.join(solverz_dir, filename)

    # Create the .Solverz directory if it does not exist
    if not os.path.exists(solverz_dir):
        os.makedirs(solverz_dir)

    # Clear the content of the file
    with open(file_path, 'w') as file:
        file.write('')
