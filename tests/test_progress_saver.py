import os
import shutil

import dill

from collab.utils import find_repo_root, progress_saver

file_path = os.path.join(
    find_repo_root(), "data", "test", "test_variable_num_0_blam_2.pkl"
)
folder_path = os.path.dirname(file_path)
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
assert not os.path.exists(folder_path)


def test_progress_saver_creation():
    num = 0
    blam = 2

    def test_var1():
        return 1

    def test_var2():
        return 2

    test_variable = progress_saver(
        name="test_variable",
        subfolder="test",
        properties=[num, blam],
        property_names=["num", "blam"],
        code_f=test_var1,
    )

    assert os.path.exists(folder_path)
    assert test_variable == 1
    with open(file_path, "rb") as f:
        loaded_variable = dill.load(f)
    assert loaded_variable == 1

    test_variable = progress_saver(
        name="test_variable",
        subfolder="test",
        properties=[num, blam],
        property_names=["num", "blam"],
        code_f=test_var2,
    )

    with open(file_path, "rb") as f:
        loaded_variable = dill.load(f)

    assert loaded_variable == 1

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
