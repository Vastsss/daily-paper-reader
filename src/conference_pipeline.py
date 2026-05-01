#!/usr/bin/env python
"""Conference retrieval pipeline wrapper.

Pipeline shape:
1. Supabase BM25 + embedding candidate retrieval.
2. RRF fusion.
3. Optional local reranker.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from conference_retrieval import build_years_token, output_paths, parse_conferences, parse_years


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
TODAY_STR = str(os.getenv("DPR_RUN_DATE") or "").strip() or datetime.now(timezone.utc).strftime("%Y%m%d")


def log(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def run_step(name: str, cmd: List[str]) -> None:
    log(f"[INFO] {name}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def load_count(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {"papers": 0, "queries": 0, "non_empty_queries": 0}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f) or {}
    queries = data.get("queries") or []
    return {
        "papers": len(data.get("papers") or []),
        "queries": len(queries),
        "non_empty_queries": sum(1 for q in queries if q.get("sim_scores")),
    }


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"[INFO] 已写入会议检索 manifest：{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="会议论文检索闭环：Supabase 召回 + RRF + 可选 rerank。")
    parser.add_argument("--config", type=str, default=str(ROOT_DIR / "config.yaml"))
    parser.add_argument("--conferences", "--conference", dest="conferences", type=str, required=True)
    parser.add_argument("--years", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=50, help="BM25 / embedding 每个查询保留候选数。")
    parser.add_argument("--rrf-top-n", type=int, default=200, help="RRF 每个查询保留候选数。")
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "archive" / TODAY_STR / "filtered"))
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embedding-device", type=str, default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--run-rerank", action="store_true", help="继续运行本地 Qwen3 reranker。")
    parser.add_argument("--rerank-top-n", type=int, default=80)
    parser.add_argument("--rerank-device", type=str, default=os.getenv("LOCAL_RERANK_DEVICE", "cpu"))
    parser.add_argument("--rerank-batch-size", type=int, default=int(os.getenv("LOCAL_RERANK_BATCH_SIZE") or "4"))
    args = parser.parse_args()

    conferences = parse_conferences(args.conferences)
    years = parse_years(args.years)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bm25_path, embedding_path = output_paths(output_dir, conferences, years)
    conf_token = "-".join(conferences)
    year_token = build_years_token(years)
    rrf_path = output_dir / f"conference-{conf_token}-{year_token}.supabase.rrf.json"
    manifest_path = output_dir / f"conference-{conf_token}-{year_token}.supabase.manifest.json"

    retrieval_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "conference_retrieval.py"),
        "--config",
        args.config,
        "--conferences",
        ",".join(conferences),
        "--years",
        ",".join(str(y) for y in years),
        "--top-k",
        str(max(int(args.top_k or 1), 1)),
        "--output-dir",
        str(output_dir),
        "--embedding-model",
        args.embedding_model,
        "--embedding-device",
        args.embedding_device,
        "--embedding-batch-size",
        str(max(int(args.embedding_batch_size or 1), 1)),
        "--embedding-max-length",
        str(max(int(args.embedding_max_length or 1), 1)),
    ]
    run_step("Conference Supabase retrieval", retrieval_cmd)

    rrf_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "2.3.retrieval_papers_rrf.py"),
        "--bm25-input",
        str(bm25_path),
        "--embedding-input",
        str(embedding_path),
        "--output",
        str(rrf_path),
        "--top-n",
        str(max(int(args.rrf_top_n or 1), 1)),
    ]
    run_step("Conference RRF", rrf_cmd)

    rerank_path = None
    if args.run_rerank:
        rank_dir = output_dir.parent / "rank" if output_dir.name == "filtered" else output_dir / "rank"
        rank_dir.mkdir(parents=True, exist_ok=True)
        rerank_path = rank_dir / f"conference-{conf_token}-{year_token}.supabase.rerank.json"
        rerank_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "3.rank_papers.py"),
            "--input",
            str(rrf_path),
            "--output",
            str(rerank_path),
            "--top-n",
            str(max(int(args.rerank_top_n or 1), 1)),
            "--rerank-device",
            args.rerank_device,
            "--rerank-batch-size",
            str(max(int(args.rerank_batch_size or 1), 1)),
        ]
        run_step("Conference rerank", rerank_cmd)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conferences": conferences,
        "years": years,
        "top_k": max(int(args.top_k or 1), 1),
        "rrf_top_n": max(int(args.rrf_top_n or 1), 1),
        "run_rerank": bool(args.run_rerank),
        "files": {
            "bm25": rel(bm25_path),
            "embedding": rel(embedding_path),
            "rrf": rel(rrf_path),
            "rerank": rel(rerank_path) if rerank_path else "",
        },
        "counts": {
            "bm25": load_count(bm25_path),
            "embedding": load_count(embedding_path),
            "rrf": load_count(rrf_path),
            "rerank": load_count(rerank_path) if rerank_path else {},
        },
    }
    write_manifest(manifest_path, manifest)
    log("[INFO] 会议论文检索闭环完成。")


if __name__ == "__main__":
    main()
