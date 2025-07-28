import kagglehub
import shutil
from pathlib import Path

# Download latest version
path = kagglehub.dataset_download("adilshamim8/exploring-mental-health-data")

print("Path to dataset files:", path)

target_dir = Path('data/')
target_dir.mkdir(exist_ok = True)
for file in Path(path).iterdir():
    shutil.copy(file, target_dir / file.name)


