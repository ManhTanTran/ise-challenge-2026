"""Build the Kaggle notebook for the expanded Kaggle Dataset layout."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Unified chunking benchmark — 62 questions / 70 evidence files

This notebook uses one attached Kaggle Dataset for data/questions, clones the benchmark code from Git, and clones the three official public repositories into Kaggle working storage.
"""),
    markdown("""## Before running

Attach the `ise-chunking-benchmark` Dataset, enable Internet and a GPU. Set `INPUT_DIR` to its Kaggle slug. The notebook clones benchmark code from `BENCHMARK_REPO`; update `BENCHMARK_REF` when pinning a commit. Add either an OpenAI key or an OpenRouter key as a Kaggle Secret named `OPENAI_API_KEY`. HiChunk's 4B vLLM phase is best run on an A100-class GPU.
"""),
    code("""from pathlib import Path
import json, os, shutil, subprocess

INPUT_DIR = Path('/kaggle/input/ise-chunking-benchmark')  # change this slug
WORK_DIR = Path('/kaggle/working/ise_chunking_benchmark')
WORK_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_REPO = 'https://github.com/ManhTanTran/ise-challenge-2026.git'
BENCHMARK_REF = 'main'  # replace with a commit SHA for a pinned run
REPO_ROOT = WORK_DIR / 'ise-challenge-2026'
if not REPO_ROOT.exists():
    subprocess.run(['git', 'clone', '--quiet', '--depth', '1', '--branch', BENCHMARK_REF, BENCHMARK_REPO, str(REPO_ROOT)], check=True)
package_source = REPO_ROOT / 'Benchmark chunking'
if not package_source.is_dir():
    raise FileNotFoundError(f'Benchmark package not found in Git checkout: {package_source}')
shutil.copytree(package_source, WORK_DIR / 'benchmark_chunking', dirs_exist_ok=True)

os.chdir(WORK_DIR)
DATASET_ROOT = INPUT_DIR

DATA_LAKE = DATASET_ROOT / 'data' / 'text_sources'
if not DATA_LAKE.exists():
    DATA_LAKE = WORK_DIR / 'data' / 'text_sources'
if not DATA_LAKE.exists():
    DATA_LAKE = next(WORK_DIR.rglob('text_sources'))
QUESTIONS = DATASET_ROOT / 'benchmark_questions.xlsx'
if not QUESTIONS.exists():
    QUESTIONS = next(INPUT_DIR.rglob('benchmark_questions.xlsx'))
PHASE1 = Path('/kaggle/working/chunking_phase1')
PHASE2 = Path('/kaggle/working/chunking_hichunk')
HICHUNK_SPLITS = Path('/kaggle/working/hichunk_splits.json')
print('Evidence files:', sum(path.is_file() for path in DATA_LAKE.rglob('*')))
"""),
    markdown("""## 0. Clone official repositories in Kaggle

The repositories are cloned into `/kaggle/working/repos`, not uploaded with the Dataset. Exact commits are checked out and recorded in `official_repos_lock.json`.
"""),
    code("""REPOS_DIR = WORK_DIR / 'repos'
REPOS_DIR.mkdir(exist_ok=True)
REPOS = {
    'late-chunking': ('https://github.com/jina-ai/late-chunking.git', '1d3bb02bf091becd0771455e4e7959463935e26c'),
    'raptor': ('https://github.com/parthsarthi03/raptor.git', '7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767'),
    'hichunk': ('https://github.com/TencentCloudADP/hichunk.git', '8d17d18931123bb4ea7b06cd394e4336085c7471'),
}
resolved = {}
for name, (url, revision) in REPOS.items():
    destination = REPOS_DIR / name
    if not destination.exists():
        subprocess.run(['git', 'clone', '--quiet', url, str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'fetch', '--quiet', '--tags'], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--quiet', revision], check=True)
    resolved[name] = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
(WORK_DIR / 'official_repos_lock.json').write_text(json.dumps(resolved, indent=2))
print(resolved)
"""),
    markdown("""## 1. Install shared dependencies
"""),
    code("""!pip -q install "transformers>=4.53,<4.57" FlagEmbedding sentence-transformers sentencepiece pymupdf python-docx python-pptx ebooklib beautifulsoup4 lxml openpyxl openai tiktoken umap-learn scikit-learn tenacity faiss-cpu
"""),
    code("""# Regression check: RAPTOR imports FaissRetriever from raptor/__init__.py.
import faiss, sys
sys.path.insert(0, str(REPOS_DIR / 'raptor'))
import raptor
print('RAPTOR dependency check passed; faiss', faiss.__version__)
"""),
    markdown("""## 2. Configure and validate GPT summaries for PIC/RAPTOR

Set `RUN_LLM_METHODS = False` to benchmark the four non-LLM methods first. An `sk-or-` key uses OpenRouter automatically; any invalid key fails here, before the benchmark starts.
"""),
    code("""from kaggle_secrets import UserSecretsClient
from openai import OpenAI

RUN_LLM_METHODS = False
try:
    API_KEY = UserSecretsClient().get_secret('OPENAI_API_KEY')
except Exception:
    API_KEY = ''

SUMMARY_MODEL = 'gpt-4o-mini'
CLIENT_ARGS = {}
if API_KEY.startswith('sk-or-'):
    CLIENT_ARGS['base_url'] = 'https://openrouter.ai/api/v1'
    SUMMARY_MODEL = 'openai/gpt-4o-mini'
if RUN_LLM_METHODS:
    if not API_KEY:
        raise RuntimeError('Missing OPENAI_API_KEY Kaggle Secret. Set RUN_LLM_METHODS=False for a no-LLM smoke run.')
    os.environ['OPENAI_API_KEY'] = API_KEY
    if 'base_url' in CLIENT_ARGS:
        os.environ['OPENAI_BASE_URL'] = CLIENT_ARGS['base_url']
    try:
        OpenAI(api_key=API_KEY, **CLIENT_ARGS).models.list()
    except Exception as exc:
        raise RuntimeError('The configured OpenAI/OpenRouter key is invalid or has no access. Fix the Kaggle Secret before running.') from exc
print({'llm_methods': RUN_LLM_METHODS, 'summary_model': SUMMARY_MODEL, 'base_url': CLIENT_ARGS.get('base_url', 'https://api.openai.com/v1')})
"""),
    markdown("""## 3. Naive Fixed + Naive Semantic

This run contains no Late Chunking and no LLM calls. Its output is isolated so a later failure cannot remove it.
"""),
    code("""PHASE_NAIVE = Path('/kaggle/working/chunking_naive')
subprocess.run([
    'python', '-X', 'utf8', '-m', 'benchmark_chunking.tools.benchmark_all_chunking',
    '--data-lake', str(DATA_LAKE), '--questions', str(QUESTIONS), '--output-dir', str(PHASE_NAIVE),
    '--methods', 'naive_fixed', 'naive_semantic', '--model', 'BAAI/bge-m3', '--device', 'cuda',
    '--window-tokens', '4096', '--source-window-chars', '8000',
], check=True)
"""),
    markdown("""## 4. Late Fixed + Late Semantic

This run is intentionally separate because token-level contextual pooling is much slower than naive chunk embedding.
"""),
    code("""PHASE_LATE = Path('/kaggle/working/chunking_late')
subprocess.run([
    'python', '-X', 'utf8', '-m', 'benchmark_chunking.tools.benchmark_all_chunking',
    '--data-lake', str(DATA_LAKE), '--questions', str(QUESTIONS), '--output-dir', str(PHASE_LATE),
    '--methods', 'late_fixed', 'late_semantic', '--model', 'BAAI/bge-m3', '--device', 'cuda',
    '--window-tokens', '4096', '--source-window-chars', '8000',
], check=True)
"""),
    markdown("""## 5. PIC and RAPTOR (optional LLM runs)

Set `RUN_LLM_METHODS = True` in the configuration cell and rerun the corresponding cell below. Each method has an independent output/cache.
"""),
    code("""PHASE_PIC = Path('/kaggle/working/chunking_pic')
if RUN_LLM_METHODS:
    subprocess.run([
        'python', '-X', 'utf8', '-m', 'benchmark_chunking.tools.benchmark_all_chunking',
        '--data-lake', str(DATA_LAKE), '--questions', str(QUESTIONS), '--output-dir', str(PHASE_PIC),
        '--methods', 'pic', '--summary-model', SUMMARY_MODEL, '--model', 'BAAI/bge-m3', '--device', 'cuda',
        '--window-tokens', '4096', '--source-window-chars', '8000',
    ], check=True)
"""),
    code("""PHASE_RAPTOR = Path('/kaggle/working/chunking_raptor')
if RUN_LLM_METHODS:
    subprocess.run([
        'python', '-X', 'utf8', '-m', 'benchmark_chunking.tools.benchmark_all_chunking',
        '--data-lake', str(DATA_LAKE), '--questions', str(QUESTIONS), '--output-dir', str(PHASE_RAPTOR),
        '--raptor-repo', str(REPOS_DIR / 'raptor'), '--methods', 'raptor_all_nodes',
        '--summary-model', SUMMARY_MODEL, '--model', 'BAAI/bge-m3', '--device', 'cuda',
        '--window-tokens', '4096', '--source-window-chars', '8000',
    ], check=True)
"""),
    markdown("""## 6. Generate and score official HiChunk

This phase is resumable per source window. If vLLM asks for a restart, restart the kernel, rerun setup through repository-cloning, then continue with this section.
"""),
    code("""!pip -q install vllm==0.10.2 nltk liger-kernel
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
"""),
    code("""subprocess.run([
    'python', '-X', 'utf8', '-m', 'benchmark_chunking.tools.generate_hichunk_splits',
    '--inputs', str(PHASE_LATE / 'hichunk_inputs.json'), '--output', str(HICHUNK_SPLITS),
    '--repo', str(REPOS_DIR / 'hichunk'), '--model', 'tencent/Youtu-HiChunk',
    '--model-deploy', 'vllm', '--window-size', '16384', '--recurrent-type', '2',
], check=True)
"""),
    code("""subprocess.run([
    'python', '-X', 'utf8', '-m', 'benchmark_chunking.tools.benchmark_all_chunking',
    '--data-lake', str(DATA_LAKE), '--questions', str(QUESTIONS), '--output-dir', str(PHASE2),
    '--raptor-repo', str(REPOS_DIR / 'raptor'), '--hichunk-splits', str(HICHUNK_SPLITS),
    '--methods', 'hichunk_flat', '--model', 'BAAI/bge-m3', '--device', 'cuda',
    '--window-tokens', '4096', '--source-window-chars', '8000',
], check=True)
"""),
    markdown("""## 5. Merge final report
"""),
    code("""summary_paths = [
    Path('/kaggle/working/chunking_naive/summary.json'),
    Path('/kaggle/working/chunking_late/summary.json'),
    Path('/kaggle/working/chunking_pic/summary.json'),
    Path('/kaggle/working/chunking_raptor/summary.json'),
    Path('/kaggle/working/chunking_hichunk/summary.json'),
]
summaries = [json.loads(path.read_text()) for path in summary_paths if path.exists()]
if not summaries:
    raise FileNotFoundError('No method summary.json files found.')
combined = {}
for summary in summaries:
    combined.update({name: value for name, value in summary.items() if name != 'config'})
base_config = next(summary['config'] for summary in summaries if 'config' in summary)
combined['config'] = {
    'questions': base_config.get('questions'),
    'documents': base_config.get('documents'),
    'dataset_contract': base_config.get('benchmark_contract'),
    'official_repos': json.loads((WORK_DIR / 'official_repos_lock.json').read_text()),
}
FINAL = Path('/kaggle/working/summary_all.json')
FINAL.write_text(json.dumps(combined, ensure_ascii=False, indent=2))
table = {name: {'File MRR': row['file_mrr'], 'Hit@1': row['chunk_hit@1'], 'Hit@8': row['chunk_hit@8']} for name, row in combined.items() if name != 'config'}
display(table)
print('Final report:', FINAL)
"""),
]

notebook = {
    "cells": cells,
    "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = Path(__file__).parents[1] / "benchmark_all_chunking_kaggle.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(target)
