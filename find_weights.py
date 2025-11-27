"""Script to find the downloaded model weights path."""

from texteller.globals import Globals
from pathlib import Path
import os

# Get HuggingFace cache directory
hf_home = os.environ.get('HF_HOME')
if not hf_home:
    hf_home = Path.home() / '.cache' / 'huggingface'
else:
    hf_home = Path(hf_home)

print(f"HuggingFace cache directory: {hf_home}")
print()

# Get repository name
repo = Globals().repo_name
print(f"Repository: {repo}")
print()

# Find model path
repo_name = repo.replace("/", "--")
model_path = hf_home / 'hub' / f'models--{repo_name}'

print(f"Model cache path: {model_path}")
print(f"Exists: {model_path.exists()}")
print()

if model_path.exists():
    snapshots_dir = model_path / 'snapshots'
    if snapshots_dir.exists():
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            print(f"Found {len(snapshots)} snapshot(s):")
            for snapshot in snapshots:
                print(f"  - {snapshot}")
                print(f"\nFor training, use this path:")
                print(f"  {snapshot}")
                
                # Show contents
                if snapshot.is_dir():
                    files = list(snapshot.iterdir())
                    print(f"\n  Contains {len(files)} files:")
                    for f in sorted(files)[:10]:  # Show first 10
                        print(f"    - {f.name}")
                    if len(files) > 10:
                        print(f"    ... and {len(files) - 10} more files")
        else:
            print("No snapshots found yet. The model may still be downloading.")
    else:
        print("Snapshots directory not found.")
else:
    print("Model not cached yet. Run the model once to download it:")
    print("  uv run texteller inference --help")
