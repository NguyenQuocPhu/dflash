from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import requests
from loguru import logger
from rich import print
from rich.table import Table
from tqdm import tqdm

random.seed(42)


CACHE_DIR = Path(__file__).parent.parent / "cache"
_GSM8K_NUMBER_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: (
            "{question}\nSolve the problem step by step. Put the reasoning inside "
            "<think>...</think>. Put exactly one number inside <answer>...</answer> "
            "(for example, <answer>160</answer>). Do not include units, words, "
            "currency symbols, equations, or any other text inside <answer>."
        ).format(**x),
        "reference": lambda x: x["answer"].split("####")[-1].strip() if "####" in x["answer"] else x["answer"],
        "cache_version": 5,
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
        "reference": lambda x: str(x["answer"]),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```".format(**x),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: x["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: x["prompt"],
        "multi_turn": True,
    },
}


def _dataset_cache_path(name: str) -> Path:
    version = DATASETS[name].get("cache_version", 2)
    return CACHE_DIR / f"{name}_v{version}.jsonl"


def _prepare_dataset(name: str) -> Path:
    from datasets import load_dataset

    cfg = DATASETS[name]
    CACHE_DIR.mkdir(exist_ok=True)
    out_path = _dataset_cache_path(name)
    tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")

    print(f"[download] {name} ...")
    dataset = load_dataset(*cfg["load_args"], **cfg["load_kwargs"])

    with open(tmp_path, "w") as f:
        for row in dataset:
            if cfg.get("multi_turn"):
                turns = cfg["format"](row)
            else:
                turns = [cfg["format"](row)]
            item = {"turns": turns}
            if "reference" in cfg:
                item["reference"] = cfg["reference"](row)
            f.write(json.dumps(item) + "\n")
    os.replace(tmp_path, out_path)

    with open(out_path) as f:
        num_samples = sum(1 for _ in f)
    print(f"[cached] {out_path}  ({num_samples} samples)")
    return out_path


def load_and_process_dataset(data_name: str) -> list[dict]:
    if data_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{data_name}'. Available: {list(DATASETS.keys())}")

    path = _dataset_cache_path(data_name)
    if not path.exists():
        _prepare_dataset(data_name)

    with open(path) as f:
        return [json.loads(line) for line in f]


def _limit_dataset(dataset: list[dict], max_samples: int | None) -> list[dict]:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    random.shuffle(dataset)
    return dataset[:max_samples]


def extract_answer(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    
    idx += len("\\boxed{")
    brace_count = 1
    for i in range(idx, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
        
        if brace_count == 0:
            return text[idx:i].strip()
    return None


def _judge_exact_match(model_output: str, reference: str) -> bool:
    ans = extract_answer(model_output)
    if ans is None:
        return False

    def normalize(s: str) -> str:
        return re.sub(r"[\s,\$]", "", s).lower()

    return normalize(ans) == normalize(reference)


@lru_cache(maxsize=None)
def _parse_reference(reference: str):
    from math_verify import parse
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    return parse(
        reference,
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
    )


def _extract_gsm8k_answer(text: str) -> str | None:
    if "<answer>" not in text or "</answer>" not in text:
        return None
    answer = text.rsplit("<answer>", 1)[-1]
    answer = answer.split("</answer>", 1)[0].strip()
    if _GSM8K_NUMBER_RE.fullmatch(answer) is None:
        return None
    return answer


def judge_correctness(model_output: str, reference: str, dataset_name: str) -> bool:
    """Grade a model response using the selected dataset's answer format."""
    if not isinstance(model_output, str) or not isinstance(reference, str):
        return False

    if dataset_name == "gsm8k":
        predicted_answer = _extract_gsm8k_answer(model_output)
        return predicted_answer is not None and predicted_answer == reference.strip()

    try:
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        gold = _parse_reference(reference)
        answer = parse(
            model_output,
            extraction_config=[
                LatexExtractionConfig(boxed_match_priority=0),
                ExprExtractionConfig(),
            ],
        )
        return bool(verify(gold, answer))
    except Exception:
        return _judge_exact_match(model_output, reference)


def _print_evaluation_sample(
    question: str,
    reference: str,
    model_output: str,
    is_correct: bool,
) -> None:
    print(f"\n{'=' * 80}")
    print(f"Question:\n{question}")
    print(f"\nReference answer:\n{reference}")
    print(f"\nLLM output:\n{model_output}")
    print(f"\nResult: {'CORRECT' if is_correct else 'INCORRECT'}")
    print(f"{'=' * 80}")


def _apply_chat_template(tokenizer, messages: list[dict], enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _make_decode_metrics(num_output_tokens: int, generation_tps: float, acceptance_lengths: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        num_output_tokens=num_output_tokens,
        time_per_output_token=1.0 / generation_tps if generation_tps > 0 else float("inf"),
        acceptance_lengths=acceptance_lengths,
    )


def _input_token_bucket(num_tokens: int) -> tuple[int, str]:
    if num_tokens < 128:
        return 0, "0-127"

    lower = 128
    while num_tokens >= lower * 2:
        lower *= 2
    return lower, f"{lower}-{lower * 2 - 1}"


def _make_speed_profile(input_tokens: int, **metrics: SimpleNamespace) -> dict:
    profile = {"input_tokens": int(input_tokens), "series": {}}
    for label, metric in metrics.items():
        output_tokens = int(metric.num_output_tokens)
        generation_time = output_tokens * float(metric.time_per_output_token)
        profile["series"][label] = (output_tokens, generation_time)
    return profile


def _print_input_speed_report(profiles: list[dict], labels: list[str]) -> None:
    if not profiles:
        return

    grouped: dict[int, dict] = {}
    for profile in profiles:
        input_tokens = int(profile["input_tokens"])
        bucket_start, bucket_label = _input_token_bucket(input_tokens)
        group = grouped.setdefault(
            bucket_start,
            {"label": bucket_label, "input_tokens": [], "series": {}},
        )
        group["input_tokens"].append(input_tokens)
        for label, (output_tokens, generation_time) in profile["series"].items():
            totals = group["series"].setdefault(label, [0, 0.0])
            if output_tokens > 0 and np.isfinite(generation_time) and generation_time > 0:
                totals[0] += output_tokens
                totals[1] += generation_time

    table = Table(title="Generation speed by input length")
    table.add_column("Input tokens", justify="right")
    table.add_column("Prompts", justify="right")
    table.add_column("Avg input", justify="right")
    for label in labels:
        table.add_column(f"{label} tok/s", justify="right")
    if labels == ["Baseline", "DFlash"]:
        table.add_column("Speedup", justify="right")

    for bucket_start in sorted(grouped):
        group = grouped[bucket_start]
        speeds: dict[str, float | None] = {}
        row = [
            group["label"],
            str(len(group["input_tokens"])),
            f"{statistics.mean(group['input_tokens']):.1f}",
        ]
        for label in labels:
            output_tokens, generation_time = group["series"].get(label, (0, 0.0))
            speed = output_tokens / generation_time if generation_time > 0 else None
            speeds[label] = speed
            row.append(f"{speed:.2f}" if speed is not None else "-")
        if labels == ["Baseline", "DFlash"]:
            baseline = speeds["Baseline"]
            dflash = speeds["DFlash"]
            speedup = dflash / baseline if baseline and dflash is not None else None
            row.append(f"{speedup:.2f}x" if speedup is not None else "-")
        table.add_row(*row)

    print(table)


def _print_decode_summary(responses: list[dict[int, SimpleNamespace]], block_size: int) -> None:
    baseline_tpot = np.mean([r[1].time_per_output_token for r in responses])
    dflash_tpot = np.mean([r[block_size].time_per_output_token for r in responses])
    print(f"Baseline throughput: {1 / baseline_tpot:.2f} tok/s")
    print(f"DFlash throughput:  {1 / dflash_tpot:.2f} tok/s")
    print(f"Decoding speedup: {baseline_tpot / dflash_tpot:.2f}")

    mean_accept = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {mean_accept:.2f}")

    acceptance_lengths = list(chain.from_iterable(r[block_size].acceptance_lengths for r in responses))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _dist_init(torch_dist) -> None:
    if "RANK" not in os.environ:
        warnings.warn("RANK not set. Skipping distributed initialization.")
        return
    torch_dist.init_process_group(backend="nccl", init_method="env://")


def _dist_size() -> int:
    return _env_int("WORLD_SIZE", 1)


def _dist_rank() -> int:
    return _env_int("RANK", 0)


def _dist_local_rank() -> int:
    return _env_int("LOCAL_RANK", 0)


def _dist_is_main() -> bool:
    return _dist_rank() == 0


def _dist_gather(torch_dist, obj: Any, dst: int = 0):
    if not torch_dist.is_initialized():
        return [obj]
    if _dist_is_main():
        objs = [None for _ in range(_dist_size())]
        torch_dist.gather_object(obj, objs, dst=dst)
        return objs
    torch_dist.gather_object(obj, dst=dst)
    return None


_TRANSFORMERS_SUPPORTED_PATTERN = re.compile(r"qwen3(?!\.5)[\w-]*|llama.*3\.1.*8b.*instruct", re.IGNORECASE)


def _check_transformers_model(model_name: str) -> None:
    if not _TRANSFORMERS_SUPPORTED_PATTERN.search(model_name):
        raise ValueError(
            f"Transformers backend does not support '{model_name}'. "
            f"Only Qwen3 series and LLaMA-3.1-8B-Instruct are supported. "
            f"Use --backend sglang or --backend vllm for other models."
        )


def _get_transformers_attn_impl() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        logger.warning(
            "flash_attn not installed. Falling back to torch.sdpa. Speedup will be lower. "
            "For optimal speedup in Transformers backend, please install: "
            "pip install flash-attn --no-build-isolation"
        )
        return "sdpa"


def _run_transformers(args: argparse.Namespace) -> None:
    import torch
    from torch import distributed as torch_dist
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .model import DFlashDraftModel, dflash_generate

    _check_transformers_model(args.model)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _dist_init(torch_dist)
    torch.cuda.set_device(_dist_local_rank())
    device = torch.device(f"cuda:{_dist_local_rank()}")
    attn_impl = _get_transformers_attn_impl()

    target = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation=attn_impl, dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_model, attn_implementation=attn_impl, dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_and_process_dataset(args.dataset)

    dataset = _limit_dataset(dataset, args.max_samples)

    responses = []
    speed_profiles = []
    correct_count = 0
    total_eval = 0
    indices = range(_dist_rank(), len(dataset), _dist_size())
    for idx in tqdm(indices, disable=not _dist_is_main()):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = _apply_chat_template(tokenizer, messages, args.enable_thinking)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            for bs in [1, block_size]:
                response[bs] = dflash_generate(
                    draft_model,
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    block_size=bs,
                    return_stats=True,
                )

            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)
            speed_profiles.append(_make_speed_profile(
                input_ids.shape[-1],
                Baseline=response[1],
                DFlash=response[block_size],
            ))

            if "reference" in instance:
                is_correct = judge_correctness(output_text, instance["reference"], args.dataset)
                if is_correct:
                    correct_count += 1
                total_eval += 1
                if args.print_samples:
                    _print_evaluation_sample(
                        user_content,
                        instance["reference"],
                        output_text,
                        is_correct,
                    )

    if _dist_size() > 1:
        responses = _dist_gather(torch_dist, responses, dst=0)
        speed_profiles = _dist_gather(torch_dist, speed_profiles, dst=0)
        eval_stats = _dist_gather(torch_dist, (correct_count, total_eval), dst=0)
        if not _dist_is_main():
            return
        responses = list(chain(*responses))
        speed_profiles = list(chain(*speed_profiles))
        correct_count = sum(s[0] for s in eval_stats)
        total_eval = sum(s[1] for s in eval_stats)

    _print_decode_summary(responses, block_size)
    _print_input_speed_report(speed_profiles, ["Baseline", "DFlash"])
    if total_eval > 0:
        print(f"Accuracy: {correct_count / total_eval * 100:.2f}% ({correct_count}/{total_eval})")


def _send_sglang(
    base_url: str,
    text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_s: int,
) -> dict:
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": text,
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    out = resp.json()
    return out if isinstance(out, dict) else out[0]


def _send_vllm(
    base_url: str,
    text: str,
    *,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_s: int,
    enable_thinking: bool = False,
) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    resp = requests.post(
        base_url + "/v1/chat/completions",
        json=body,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json()


def _run_mlx(args: argparse.Namespace) -> None:
    from mlx_lm import stream_generate as stream_generate_baseline
    from mlx_lm.sample_utils import make_sampler

    from .model_mlx import load, load_draft, stream_generate

    sampler = make_sampler(temp=args.temperature)

    logger.info(f"Loading target: {args.model}")
    model, tokenizer = load(args.model)
    logger.info(f"Loading draft: {args.draft_model}")
    draft = load_draft(args.draft_model)
    block_size = args.block_size if args.block_size is not None else int(draft.config.block_size)

    dataset = load_and_process_dataset(args.dataset)
    dataset = _limit_dataset(dataset, args.max_samples)

    warmup_prompt = tokenizer.encode("Hi")
    list(stream_generate_baseline(model, tokenizer, warmup_prompt, 3, sampler=sampler))
    list(stream_generate(model, draft, tokenizer, warmup_prompt, block_size, 3, sampler=sampler))

    responses = []
    speed_profiles = []
    correct_count = 0
    total_eval = 0
    for idx in tqdm(range(len(dataset))):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            prompt = _apply_chat_template(tokenizer, messages, args.enable_thinking)

            response = {}

            tokens_bl, tps_bl = [], 0
            for r in stream_generate_baseline(model, tokenizer, prompt, args.max_new_tokens, sampler=sampler):
                tokens_bl.append(r.token)
                tps_bl = r.generation_tps
            response[1] = _make_decode_metrics(len(tokens_bl), tps_bl, [1])

            tokens_df, accs, tps_df = [], [], 0
            for r in stream_generate(model, draft, tokenizer, prompt, block_size, args.max_new_tokens, sampler=sampler):
                tokens_df.extend(r.tokens)
                accs.append(r.accepted)
                tps_df = r.generation_tps
            response[block_size] = _make_decode_metrics(len(tokens_df), tps_df, accs)

            output_text = tokenizer.decode(tokens_df)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)
            speed_profiles.append(_make_speed_profile(
                len(tokenizer.encode(prompt)),
                Baseline=response[1],
                DFlash=response[block_size],
            ))

            if "reference" in instance:
                is_correct = judge_correctness(output_text, instance["reference"], args.dataset)
                if is_correct:
                    correct_count += 1
                total_eval += 1
                if args.print_samples:
                    _print_evaluation_sample(
                        user_content,
                        instance["reference"],
                        output_text,
                        is_correct,
                    )

    _print_decode_summary(responses, block_size)
    _print_input_speed_report(speed_profiles, ["Baseline", "DFlash"])
    if total_eval > 0:
        print(f"Accuracy: {correct_count / total_eval * 100:.2f}% ({correct_count}/{total_eval})")


def _run_server(args: argparse.Namespace) -> None:
    is_vllm = args.backend == "vllm"
    dataset = load_and_process_dataset(args.dataset)
    tokenizer = None
    if not is_vllm:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    num_prompts = args.num_prompts + args.concurrency
    prompts_with_refs = []
    for i in range(num_prompts):
        item = dataset[i % len(dataset)]
        user_content = item["turns"][0]
        ref = item.get("reference")
        
        if is_vllm:
            prompts_with_refs.append((user_content, ref, user_content, None))
        else:
            formatted_prompt = _apply_chat_template(
                tokenizer,
                [{"role": "user", "content": user_content}],
                args.enable_thinking,
            )
            prompts_with_refs.append((
                formatted_prompt,
                ref,
                user_content,
                len(tokenizer.encode(formatted_prompt)),
            ))

    def send_one(prompt: str) -> dict:
        if is_vllm:
            return _send_vllm(
                args.base_url,
                prompt,
                model=args.model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                timeout_s=args.timeout_s,
                enable_thinking=args.enable_thinking,
            )
        return _send_sglang(
            args.base_url,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            timeout_s=args.timeout_s,
        )

    def send_one_timed(prompt: str) -> tuple[dict, float]:
        request_start = time.perf_counter()
        out = send_one(prompt)
        return out, time.perf_counter() - request_start

    if not is_vllm:
        try:
            requests.get(args.base_url + "/flush_cache", timeout=60).raise_for_status()
        except Exception:
            print("Warning: /flush_cache failed. Continuing.")

    bs = max(args.concurrency, 1)
    if len(prompts_with_refs) > bs:
        print(f"[warmup] {bs} requests ...")
        with ThreadPoolExecutor(max_workers=bs) as pool:
            list(pool.map(send_one, [p[0] for p in prompts_with_refs[:bs]]))
        prompts_with_refs = prompts_with_refs[bs:]

    print(f"Running benchmark: {args.num_prompts} prompts, concurrency={args.concurrency} ...")
    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []
    correct_count = 0
    total_eval = 0
    speed_profiles = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(send_one_timed, p[0]): (p[1], p[2], p[3])
            for p in prompts_with_refs
        }
        for fut in tqdm(as_completed(futures), total=len(prompts_with_refs), desc="Benchmarking"):
            ref, question, estimated_input_tokens = futures[fut]
            out, request_latency = fut.result()
            
            output_text = ""
            if is_vllm:
                if "choices" in out and len(out["choices"]) > 0:
                    output_text = out["choices"][0].get("message", {}).get("content", "")
            else:
                output_text = out.get("text", "")
            
            if ref is not None:
                is_correct = judge_correctness(output_text, ref, args.dataset)
                if is_correct:
                    correct_count += 1
                total_eval += 1
                if args.print_samples:
                    _print_evaluation_sample(question, ref, output_text, is_correct)
            if is_vllm:
                usage = out.get("usage", {})
                completion_tokens = int(usage.get("completion_tokens", 0))
                input_tokens = int(usage.get("prompt_tokens", 0) or estimated_input_tokens or 0)
                total_tokens += completion_tokens
            else:
                meta = out.get("meta_info", {}) or {}
                completion_tokens = int(meta.get("completion_tokens", 0))
                input_tokens = int(meta.get("prompt_tokens", 0) or estimated_input_tokens or 0)
                total_tokens += completion_tokens
                spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
                if "spec_accept_length" in meta:
                    try:
                        spec_accept_lengths.append(float(meta["spec_accept_length"]))
                    except (TypeError, ValueError):
                        pass
            if input_tokens > 0 and completion_tokens > 0:
                speed_profiles.append({
                    "input_tokens": input_tokens,
                    "series": {args.backend: (completion_tokens, request_latency)},
                })

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    print(f"\n{'=' * 50}")
    print(f"Backend:          {args.backend}")
    print(f"Dataset:          {args.dataset}")
    print(f"Num prompts:      {args.num_prompts}")
    print(f"Concurrency:      {args.concurrency}")
    print(f"Latency:          {latency:.1f}s")
    print(f"Output tokens:    {total_tokens}")
    print(f"Throughput:       {toks_per_s:,.2f} tok/s")
    if spec_accept_lengths:
        print(f"Accept length:    {statistics.mean(spec_accept_lengths):.3f}")
    if spec_verify_ct_sum > 0:
        print(f"Spec verify ct:   {spec_verify_ct_sum}")
    if total_eval > 0:
        print(f"Accuracy:         {correct_count / total_eval * 100:.2f}% ({correct_count}/{total_eval})")
    print(f"{'=' * 50}")
    _print_input_speed_report(speed_profiles, [args.backend])


def main() -> None:
    parser = argparse.ArgumentParser(description="DFlash benchmark")
    parser.add_argument("--backend", choices=["transformers", "sglang", "vllm", "mlx"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--num-prompts", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--print-samples", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=3600)

    args = parser.parse_args()

    assert not (args.enable_thinking and any(x in args.model.lower() for x in ["qwen3-4b", "qwen3-8b"])), (
        "DFlash draft models for Qwen3-4B and Qwen3-8B were not trained with thinking traces. "
        "Using --enable-thinking will lead to suboptimal performance."
    )

    if args.backend == "transformers":
        if args.draft_model is None:
            parser.error("--draft-model is required for transformers backend")
        _run_transformers(args)
    elif args.backend == "mlx":
        if args.draft_model is None:
            parser.error("--draft-model is required for mlx backend")
        _run_mlx(args)
    else:
        _run_server(args)


if __name__ == "__main__":
    main()
