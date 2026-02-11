import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


list_of_files = [
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "research/__init__.py",
    "src/utils/__init__.py",
    "src/utils/common.py",
    "src/security/__init__.py",
    "config/config.py",
    "main.py",
    "Dockerfile",
    ".env",
    "config.properties",
    "requirements.txt",
    "setup.py",
    "test.py",
    "ui.py"
]



if __name__ == "__main__":
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)


        if filedir !="":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory; {filedir} for the file: {filename}")

        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w") as f:
                pass
                logging.info(f"Creating empty file: {filepath}")
        else:
            logging.info(f"{filename} is already exists")