import os

import dill

from collab.utils import find_repo_root


class ProgressSaverContext:
    def __init__(self, file_path):
        self.file_path = file_path

    def save_progress(self, name):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "wb") as f:
            dill.dump(name, f)

    def load_progress(self):
        with open(self.file_path, "rb") as f:
            return dill.load(f)


def progress_saver(
    name, subfolder, filename=None, properties=None, property_names=None, code_f=None, force_rerun=False
):
    properties_string = "_".join(
        [f"{var_name}_{var}" for var_name, var in zip(property_names, properties)]
    )
    if filename is None:
        filename = f"{name}_{properties_string}.pkl"

    file_path = os.path.join(find_repo_root(), "data", subfolder, filename)

    print(file_path)

    progress_context = ProgressSaverContext(file_path)

    if not os.path.exists(file_path) or force_rerun:
        print("executing code")
        try:
            result = code_f()
            progress_context.save_progress(result)
            return result
        except Exception as e:
            print(f"Error executing code: {e}")

    else:
        print("path found, loading data")
        loaded_data = progress_context.load_progress()
        return loaded_data
