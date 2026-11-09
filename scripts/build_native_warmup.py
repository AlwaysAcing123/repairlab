"""Build a finite supervised native-tool warm-up notebook, without credentials."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repairlab.native_tools import tools  # noqa: E402
from repairlab.revision import MODEL, REVISION  # noqa: E402
from repairlab.tool_warmup import bundle  # noqa: E402


def build(destination):
    data = {
        "bundle": bundle(),
        "tools": tools(),
        "model": MODEL,
        "model_revision": REVISION,
        "expected_original_base_sha256": "2d4bad117c2ae98f4a69740ac05d48c237237d84711349a396c5e98866553204",
    }
    texts = [
        (
            "markdown",
            "# RepairLab supervised native-tool warm-up\n\n"
            "Original-model evaluation, 60 supervised LoRA steps on 64 next-action examples from eight new training programs, "
            "then the same evaluation with the adapter. Four other programs supply 16 separate development probes; "
            "the 12 historical diagnostic probes are repeated separately. These are interface checks, not autonomous repair scores. "
            "No RL, paid inference, account credentials, or model-generated-code execution. "
            "Prompt/history tokens are masked from training loss. The final checkpoint is fixed in advance; no selection based on evaluation scores.",
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
            "import json\nPAYLOAD=json.loads(" + repr(json.dumps(data, ensure_ascii=False)) + ")\n",
        ),
        ("code", (ROOT / "scripts" / "native_warmup_runtime.py").read_text()),
        (
            "code",
            "try:\n    comparison=run(PAYLOAD)\nexcept Exception as error:\n"
            "    import json,shutil\n    from pathlib import Path\n"
            "    folders=sorted(Path('.').glob('repairlab-native-warmup-*'))\n"
            "    folders=[p for p in folders if p.is_dir()]\n"
            "    if folders:\n        folder=folders[-1]\n"
            "        (folder/'failure.json').write_text(json.dumps({'error':type(error).__name__+': '+str(error)[:1000]}))\n"
            "        shutil.make_archive(str(folder),'zip',folder)\n    raise\n",
        ),
    ]
    cells = []
    for index, (kind, text) in enumerate(texts):
        cell = {"cell_type": kind, "metadata": {}, "id": f"warmup-{index:02d}", "source": text}
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
