"""Build an inference-only native repair round using private saved-model outputs."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(destination, batch):
    texts = [
        (
            "markdown",
            "# RepairLab native autonomous repair pilot\n\n"
            "One next action per unfinished episode. Original model versus the exact saved supervised adapter, "
            "with identical native tools and inference budgets. Model output is never executed on this GPU. "
            "Only public task conversations and checkpoint hashes are embedded; private checks and account "
            "credentials stay local. The adapter is loaded from the attached private warm-up notebook output. "
            "No training occurs. Full parameter fingerprints are checked before and after inference.",
        ),
        (
            "code",
            "import os,sys,subprocess\nos.environ['CUDA_VISIBLE_DEVICES']='0'\n"
            "os.environ['TOKENIZERS_PARALLELISM']='false'\nos.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'\n"
            "subprocess.check_call([sys.executable,'-m','pip','install','transformers==5.2.0','trl==1.8.0','peft==0.18.1'])\n"
            "import torch\nif not torch.cuda.is_available():\n    raise RuntimeError('Select a GPU runtime')\n",
        ),
        (
            "code",
            "import json\nBATCH=json.loads(" + repr(json.dumps(batch, ensure_ascii=False)) + ")\n",
        ),
        ("code", (ROOT / "scripts" / "native_repair_runtime.py").read_text()),
        (
            "code",
            "try:\n    run_native_round(BATCH)\nexcept Exception:\n"
            "    import traceback, shutil\n    from pathlib import Path\n"
            "    for folder in Path('.').glob('repairlab-native-pilot-round-*'):\n"
            "        if folder.is_dir():\n"
            "            (folder/'failure.txt').write_text(traceback.format_exc())\n"
            "            shutil.make_archive(str(folder), 'zip', folder)\n    raise\n",
        ),
    ]
    cells = []
    for index, (kind, text) in enumerate(texts):
        cell = {
            "cell_type": kind,
            "metadata": {},
            "id": f"native-repair-{index:02d}",
            "source": text,
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "kaggle": {"isInternetEnabled": True, "isGpuEnabled": True},
        },
        "cells": cells,
    }
    Path(destination).write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")
    return notebook
