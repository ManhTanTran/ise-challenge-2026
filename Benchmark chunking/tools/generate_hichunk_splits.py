"""Generate resumable HiChunk boundaries for benchmark source windows."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path


def _load_official_module(repo: Path):
    module_path = repo / "pipeline" / "chunking" / "HiChunk" / "HiChunk.py"
    source = module_path.read_text(encoding="utf-8")
    scheduler_original = "max_num_batched_tokens=window_size+1000, tensor_parallel_size=1,"
    scheduler_patched = (
        "max_num_batched_tokens=min(32768, window_size + max_new_token), "
        "max_model_len=min(32768, window_size + max_new_token), tensor_parallel_size=1,"
    )
    if scheduler_original not in source and scheduler_patched not in source:
        raise RuntimeError("HiChunk scheduler compatibility patch did not match the pinned official revision.")
    patched = source.replace("use_fast=False", "use_fast=True").replace(
        scheduler_original, scheduler_patched
    )
    if patched != source:
        module_path.write_text(patched, encoding="utf-8")
    module_dir = str(module_path.parent.resolve())
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("official_hichunk", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official HiChunk module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run(args: argparse.Namespace) -> dict:
    inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    output_path = Path(args.output)
    results = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    official = _load_official_module(Path(args.repo))
    engine = official.InferenceEngine(
        args.model,
        args.window_size,
        args.model_deploy,
        max_new_token=args.max_new_tokens,
    )
    for index, (key, item) in enumerate(inputs.items(), 1):
        if key in results and results[key].get("splits"):
            continue
        print(f"[{index}/{len(inputs)}] {key}", flush=True)
        result = await engine.iterative_inf(
            official.PROMPT,
            item["text"],
            limit=args.limit,
            recurrent_type=args.recurrent_type,
        )
        result["relative_path"] = item["relative_path"]
        result["window_index"] = item["window_index"]
        results[key] = result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", default="tencent/Youtu-HiChunk")
    parser.add_argument("--model-deploy", default="vllm", help="vllm or an OpenAI-compatible ip:port")
    parser.add_argument("--window-size", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--recurrent-type", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    results = asyncio.run(run(parse_args()))
    print(f"Completed HiChunk windows: {len(results)}")


if __name__ == "__main__":
    main()
