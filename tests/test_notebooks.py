# import glob
# import os
# import subprocess

# from collab.utils import find_repo_root

# root = find_repo_root()

# os.environ["CI"] = "1"  # possibly redundant


# def test_notebooks_communicators():
#     notebook_path = f"{root}/docs/foraging/communicators"
#     notebooks = glob.glob(os.path.join(notebook_path, "*.ipynb"))

#     # run this command from terminal if the test fails to identify the source of trouble, if any
#     pytest_command = f"C1=1 python -m pytest --nbval-lax --dist loadscope -n auto {' '.join(notebooks)}"

#     subprocess.run(pytest_command, shell=True, check=True)
