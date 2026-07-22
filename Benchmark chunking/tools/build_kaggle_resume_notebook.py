"""Build a Kaggle notebook that resumes PIC, RAPTOR, and HiChunk only."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Resume chunking benchmark — PIC, RAPTOR, HiChunk

Use this notebook after Naive/Late have completed. It does not re-run them.
For HiChunk, `/kaggle/working/chunking_late/hichunk_inputs.json` must still be
present in the same Kaggle session. Save that file as an output before starting
a fresh session.
"""),
    code("""from pathlib import Path
import json, os, shutil, subprocess

MOUNT_ROOT = Path('/kaggle/input')
DATA_LAKE = next((path for path in MOUNT_ROOT.rglob('text_sources') if path.is_dir()), None)
QUESTIONS = next((path for path in MOUNT_ROOT.rglob('benchmark_questions.xlsx') if path.is_file()), None)
if DATA_LAKE is None or QUESTIONS is None:
    raise FileNotFoundError(f'Missing data/text_sources or benchmark_questions.xlsx under {MOUNT_ROOT}')

WORK_DIR = Path('/kaggle/working/ise_chunking_benchmark')
WORK_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_REPO = 'https://github.com/ManhTanTran/ise-challenge-2026.git'
REPO_ROOT = WORK_DIR / 'ise-challenge-2026'
if not REPO_ROOT.exists():
    subprocess.run(['git', 'clone', '--quiet', '--depth', '1', BENCHMARK_REPO, str(REPO_ROOT)], check=True)
shutil.copytree(REPO_ROOT / 'Benchmark chunking', WORK_DIR / 'benchmark_chunking', dirs_exist_ok=True)
os.chdir(WORK_DIR)

PHASE_NAIVE = Path('/kaggle/working/chunking_naive')
PHASE_LATE = Path('/kaggle/working/chunking_late')
PHASE_PIC = Path('/kaggle/working/chunking_pic')
PHASE_RAPTOR = Path('/kaggle/working/chunking_raptor')
PHASE_HICHUNK = Path('/kaggle/working/chunking_hichunk')
HICHUNK_INPUTS = PHASE_LATE / 'hichunk_inputs.json'
HICHUNK_SPLITS = Path('/kaggle/working/hichunk_splits.json')
print({'data_lake': str(DATA_LAKE), 'questions': str(QUESTIONS), 'hichunk_inputs_exists': HICHUNK_INPUTS.exists()})
"""),
    code("""REPOS_DIR = WORK_DIR / 'repos'
REPOS_DIR.mkdir(exist_ok=True)
REPOS = {
    'raptor': ('https://github.com/parthsarthi03/raptor.git', '7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767'),
    'hichunk': ('https://github.com/TencentCloudADP/hichunk.git', '8d17d18931123bb4ea7b06cd394e4336085c7471'),
}
for name, (url, revision) in REPOS.items():
    destination = REPOS_DIR / name
    if not destination.exists():
        subprocess.run(['git', 'clone', '--quiet', url, str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'fetch', '--quiet', '--tags'], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--quiet', revision], check=True)
"""),
    markdown("""## PIC and RAPTOR dependencies
"""),
    code("""!pip -q install "transformers==4.53.0" FlagEmbedding sentence-transformers sentencepiece pymupdf python-docx python-pptx ebooklib beautifulsoup4 lxml openpyxl openai tiktoken umap-learn scikit-learn tenacity faiss-cpu
"""),
    code("""from kaggle_secrets import UserSecretsClient
from openai import OpenAI

API_KEY = UserSecretsClient().get_secret('OPENAI_API_KEY')
if not API_KEY:
    raise RuntimeError('Set OPENAI_API_KEY Kaggle Secret before running PIC/RAPTOR.')
SUMMARY_MODEL = 'gpt-4o-mini'
CLIENT_ARGS = {}
if API_KEY.startswith('sk-or-'):
    CLIENT_ARGS['base_url'] = 'https://openrouter.ai/api/v1'
    SUMMARY_MODEL = 'openai/gpt-4o-mini'
os.environ['OPENAI_API_KEY'] = API_KEY
if 'base_url' in CLIENT_ARGS:
    os.environ['OPENAI_BASE_URL'] = CLIENT_ARGS['base_url']
OpenAI(api_key=API_KEY, **CLIENT_ARGS).models.list()
print({'summary_model': SUMMARY_MODEL, 'base_url': CLIENT_ARGS.get('base_url', 'https://api.openai.com/v1')})
"""),
    markdown("""## Run PIC
"""),
    code("""subprocess.run([
    'python', '-X', 'utf8', '-m', 'benchmark_chunking.tools.benchmark_all_chunking',
    '--data-lake', str(DATA_LAKE), '--questions', str(QUESTIONS), '--output-dir', str(PHASE_PIC),
    '--methods', 'pic', '--summary-model', SUMMARY_MODEL, '--model', 'BAAI/bge-m3', '--device', 'cuda',
    '--window-tokens', '4096', '--source-window-chars', '8000',
], check=True)
"""),
    markdown("""## Run RAPTOR
"""),
    code("""subprocess.run([
    'python', '-X', 'utf8', '-m', 'benchmark_chunking.tools.benchmark_all_chunking',
    '--data-lake', str(DATA_LAKE), '--questions', str(QUESTIONS), '--output-dir', str(PHASE_RAPTOR),
    '--raptor-repo', str(REPOS_DIR / 'raptor'), '--methods', 'raptor_all_nodes',
    '--summary-model', SUMMARY_MODEL, '--model', 'BAAI/bge-m3', '--device', 'cuda',
    '--window-tokens', '4096', '--source-window-chars', '8000',
], check=True)
"""),
    markdown("""## HiChunk dependencies and preflight

These are the versions specified by the HiChunk model card. Restart the Kaggle
kernel if this installation replaces active Torch/Transformers packages; the
files under `/kaggle/working` remain available after a kernel restart.
"""),
    code("""!pip -q install torch==2.7.0 vllm==0.9.1 transformers==4.53.0 nltk liger-kernel
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
"""),
    code("""from transformers import AutoTokenizer
if not HICHUNK_INPUTS.exists():
    raise FileNotFoundError(f'Missing {HICHUNK_INPUTS}. Restore it from the completed Late run before running HiChunk.')
tokenizer = AutoTokenizer.from_pretrained('tencent/Youtu-HiChunk', trust_remote_code=True, use_fast=False)
if isinstance(tokenizer, bool):
    raise RuntimeError('HiChunk tokenizer is invalid. Restart kernel after installing the pinned dependency stack.')
print('HiChunk tokenizer:', tokenizer.__class__.__name__)
"""),
    markdown("""## Generate and score HiChunk
"""),
    code("""subprocess.run([
    'python', '-X', 'utf8', '-m', 'benchmark_chunking.tools.generate_hichunk_splits',
    '--inputs', str(HICHUNK_INPUTS), '--output', str(HICHUNK_SPLITS),
    '--repo', str(REPOS_DIR / 'hichunk'), '--model', 'tencent/Youtu-HiChunk',
    '--model-deploy', 'vllm', '--window-size', '16384', '--recurrent-type', '2',
], check=True)
"""),
    code("""subprocess.run([
    'python', '-X', 'utf8', '-m', 'benchmark_chunking.tools.benchmark_all_chunking',
    '--data-lake', str(DATA_LAKE), '--questions', str(QUESTIONS), '--output-dir', str(PHASE_HICHUNK),
    '--raptor-repo', str(REPOS_DIR / 'raptor'), '--hichunk-splits', str(HICHUNK_SPLITS),
    '--methods', 'hichunk_flat', '--model', 'BAAI/bge-m3', '--device', 'cuda',
    '--window-tokens', '4096', '--source-window-chars', '8000',
], check=True)
"""),
    markdown("""## Merge available results
"""),
    code("""summary_paths = [
    PHASE_NAIVE / 'summary.json', PHASE_LATE / 'summary.json', PHASE_PIC / 'summary.json',
    PHASE_RAPTOR / 'summary.json', PHASE_HICHUNK / 'summary.json',
]
summaries = [json.loads(path.read_text()) for path in summary_paths if path.exists()]
combined = {}
for summary in summaries:
    combined.update({name: value for name, value in summary.items() if name != 'config'})
Path('/kaggle/working/summary_all.json').write_text(json.dumps(combined, ensure_ascii=False, indent=2))
import pandas as pd
display(pd.DataFrame([
    {'method': name, 'file_mrr': value.get('file_mrr'), 'recall@5': value.get('file_recall@5'), 'fully@5': value.get('file_fully_covered@5')}
    for name, value in combined.items()
]).sort_values('file_mrr', ascending=False))
"""),
]

notebook = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = Path(__file__).parents[1] / "run_pic_raptor_hichunk_kaggle.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(target)
