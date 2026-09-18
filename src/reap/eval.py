import logging
from typing import Tuple
import pathlib
import os
import json
import shutil
import subprocess
import threading
import time
import requests
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from lm_eval import evaluator
from lm_eval.utils import make_table
from evalplus.evaluate import evaluate as evalplus_evaluator
from vllm.entrypoints.openai.api_server import run_server
from vllm.engine.arg_utils import AsyncEngineArgs
import uvloop

from reap.args import ReapArgs, ModelArgs, EvalArgs
from reap.model_util import patched_model_map, MODEL_ATTRS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_original_model_name(model_name: str) -> Tuple[str, bool]:
    original_model_name_map = {
        "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
        "Llama-4-Scout-17B-16E-Instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "ERNIE-4.5-21B-A3B-PT": "baidu/ERNIE-4.5-21B-A3B-PT",
        "DeepSeek-V2-Lite-Chat": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "Qwen3-Coder-30B-A3B-Instruct": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen3-30B-A3B-Instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "GLM-4.5-Air": "zai-org/GLM-4.5-Air",
        "Qwen3-Coder-480B-A35B-Instruct-FP8": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    }

    original_model = None
    for key, value in original_model_name_map.items():
        if key in model_name:
            original_model = value
            break
    uncompressed_model = False
    if original_model is None:
        # it's an uncompressed model or bad path
        if model_name in original_model_name_map.values():
            original_model = model_name
            uncompressed_model = True
        else:
            logger.warning(
                f"Could not find original model for {model_name}, using model_name as original_model"
            )
            original_model = model_name
    return original_model, uncompressed_model


def wait_for_server(base_url, timeout=2400, check_interval=5):
    """Wait for the server to be ready by checking the health endpoint."""
    health_url = f"{base_url}/health"
    start_time = time.time()

    logger.info(f"Waiting for server to be ready at {health_url}")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                logger.info("Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        logger.info(f"Server not ready yet, waiting {check_interval} seconds...")
        time.sleep(check_interval)

    raise TimeoutError(f"Server did not become ready within {timeout} seconds")


def start_server_watchdog(process, base_url=None, server_log_path=None,
                          poll_interval=30, max_unhealthy_probes=10,
                          log_stale_secs=600):
    """Fail-fast guard against silent retries on a dead OR hung server.

    The eval clients (lm-eval / EvalPlus / evalscope) retry failed HTTP calls
    internally for hours. If the vLLM server dies (GPU/NCCL worker fault) the
    retries hammer a dead endpoint indefinitely. Three distinct failure modes,
    each with its own detector:

    * the server *process exits*    -> caught by ``process.poll()``;
    * the *front-end* stops serving -> caught by probing ``/health``;
    * the *engine core deadlocks*   -> the process is alive AND ``/health`` still
      returns 200 (the API front-end is fine) but the engine stops making
      progress. This is the observed multi-hour zombie (client stuck retrying
      /v1/completions with 0 generation throughput). ``/health`` cannot see it.
      Detected here by watching the **server log mtime**: vLLM emits a metrics
      heartbeat every ~10s, so if the log has not been written for
      ``log_stale_secs`` the engine is wedged.

    Hard-aborts the whole job the moment the server is unrecoverable, so SLURM
    marks it FAILED in ~minutes instead of burning a GPU node for hours.
    """
    health_url = f"{base_url}/health" if base_url else None

    def _watch():
        unhealthy = 0
        while True:
            ret = process.poll()
            if ret is not None:
                logger.error(
                    "vLLM server process exited unexpectedly (code=%s). "
                    "Aborting eval job to avoid silent retries on a dead server.",
                    ret,
                )
                os._exit(42)
            # Active liveness probe: a hung front-end keeps the process alive but
            # stops answering /health. Require several consecutive failures so a
            # transient blip (busy server, brief GC) does not trip the guard.
            if health_url is not None:
                try:
                    r = requests.get(health_url, timeout=10)
                    unhealthy = 0 if r.status_code == 200 else unhealthy + 1
                except requests.exceptions.RequestException:
                    unhealthy += 1
                if unhealthy >= max_unhealthy_probes:
                    logger.error(
                        "vLLM server failed /health %d consecutive times "
                        "(process alive but hung). Aborting eval job.",
                        unhealthy,
                    )
                    os._exit(43)
            # Engine-progress probe: /health can return 200 while the engine core
            # is deadlocked. vLLM writes a metrics heartbeat continuously, so a
            # stale server log means the engine is wedged.
            if server_log_path is not None and os.path.exists(server_log_path):
                age = time.time() - os.path.getmtime(server_log_path)
                if age > log_stale_secs:
                    logger.error(
                        "vLLM server log '%s' has been stale for %.0fs "
                        "(engine core wedged; /health may still pass). "
                        "Aborting eval job.",
                        server_log_path,
                        age,
                    )
                    os._exit(44)
            time.sleep(poll_interval)

    t = threading.Thread(target=_watch, name="server-watchdog", daemon=True)
    t.start()
    logger.info(
        "Started server watchdog (poll %ss, health-abort after %d misses, "
        "log-stale-abort after %ds)",
        poll_interval,
        max_unhealthy_probes,
        log_stale_secs,
    )
    return t


def start_server(model_name, model_args, eval_args, seed, log_file, port):
    """Starts a VLLM server for the specified model."""

    num_gpus = torch.cuda.device_count()
    logger.info("Running on %d GPUs", num_gpus)

    override_generation_config = {}
    if not eval_args.greedy:
        override_generation_config = {
            "temperature": eval_args.temperature,
            "top_p": eval_args.top_p,
            "top_k": eval_args.top_k,
            "min_p": eval_args.min_p,
        }
        logger.info(
            "Using sampling with temperature=%s, top_p=%s, top_k=%s, min_p=%s",
            eval_args.temperature,
            eval_args.top_p,
            eval_args.top_k,
            eval_args.min_p,
        )
    else:
        logger.info("Using greedy decoding")
    override_generation_config_str = json.dumps(override_generation_config)

    hf_overrides = {}
    max_num_seqs = int(os.environ.get("VLLM_MAX_NUM_SEQS", "32"))
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384"))
    gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85"))
    if model_args.num_experts_per_tok_override is not None:
        logger.info(
            f"Overriding number of experts per token to {model_args.num_experts_per_tok_override}"
        )
        key = "num_experts_per_tok"
        if "ernie" in model_name.lower():
            key = "moe_k"
        hf_overrides = {key: model_args.num_experts_per_tok_override}
    hf_overrides_str = json.dumps(hf_overrides)
    max_num_batched_tokens = int(
        os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "4096")
    )

    original_model_name, _ = get_original_model_name(model_name)

    # TODO: once  limit_mm_per_prompt={"image": 1 if use_image else 0}, is stable, use
    # in place of patching vllm.model_executor.models.registry for Llama4.

    server_command = [
        "vllm",
        "serve",
        model_name,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--tensor-parallel-size",
        str(num_gpus),
        "--seed",
        str(seed),
        "--port",
        str(port),
        "--enable-expert-parallel",
        "--disable-custom-all-reduce",
        "--enable-chunked-prefill",
        "--disable-log-requests",
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--override-generation-config",
        override_generation_config_str,
        "--trust-remote-code",
        "--hf-overrides",
        hf_overrides_str,
        "--served-model-name",  # for HELM eval.
        original_model_name,
        model_name,
    ]

    logger.info(f"Starting VLLM OpenAI API server for {model_name} on port {port}")
    logger.info(
        f"Using {num_gpus} GPUs with {gpu_memory_utilization} memory utilization"
    )
    logger.info(f"Command: {' '.join(server_command)}")

    # Start the server process with log redirection
    with open(log_file, "w") as log_file:
        process = subprocess.Popen(
            server_command, stdout=log_file, stderr=subprocess.STDOUT
        )

    # Connect over loopback (127.0.0.1), NOT 0.0.0.0. The server binds 0.0.0.0
    # (all interfaces) but 0.0.0.0 is a listen address, not a valid connect
    # target: some HTTP clients (e.g. evalscope's, used by the math phase)
    # fail to connect to it and retry forever, while others (EvalPlus) tolerate
    # it. 127.0.0.1 reaches the 0.0.0.0-bound server reliably for all clients.
    base_url = f"http://127.0.0.1:{port}"

    wait_for_server(base_url)

    return base_url, process


def run_evaluate(model_args, results_dir, eval_args, seed):
    model_name = model_args.model_name
    if isinstance(model_name, pathlib.Path):
        model_name = model_name.__str__()
    # The OpenAI-compatible clients (EvalPlus, evalscope math) send an
    # `Authorization: Bearer <OPENAI_API_KEY>` header. Our local vLLM server does
    # not require a key, but an EMPTY key produces the illegal header value
    # `Bearer ` which httpx/h11 rejects, surfacing as a misleading
    # `APIConnectionError` that the EvalPlus fork retries forever. `os.getenv(...,
    # "none")` does NOT help because the var is set-but-empty (not unset). Force a
    # non-empty placeholder so the header is well-formed (vLLM ignores its value).
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "none"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    use_server = eval_args.use_server
    if results_dir is None:
        model_short_name = model_name.split("/")[-1]
        if model_args.num_experts_per_tok_override is not None:
            model_short_name += (
                f"-num_experts_per_tok_{model_args.num_experts_per_tok_override}"
            )
        results_dir = pathlib.Path.cwd() / "artifacts" / "eval" / model_short_name
    if isinstance(results_dir, str):
        results_dir = pathlib.Path(results_dir)
    if not eval_args.greedy:
        results_dir = results_dir.parent / f"{results_dir.name}_sampling"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {results_dir}")
    num_gpus = torch.cuda.device_count()
    model_name = patched_model_map(model_name)
    if use_server:
        server_endpoint, process = start_server(
            model_name,
            model_args,
            eval_args,
            seed,
            log_file=eval_args.server_log_file_name,
            port=eval_args.vllm_port,
        )
        start_server_watchdog(
            process,
            base_url=server_endpoint,
            server_log_path=eval_args.server_log_file_name,
        )

    if eval_args.run_lm_eval:
        results_file_base_name = results_dir / "lm_eval_results"
        model_args = {
            "pretrained": model_name,
            "tensor_parallel_size": num_gpus,
            "gpu_memory_utilization": 0.85,
            "num_concurrent": 32,
            "timeout": 2400,
            "max_retries": 10,
            "trust_remote_code": True,
        }
        if "baidu" in model_name.lower():
            logger.warning("Using slow tokenizer for Ernie-4.5")
            model_args["use_fast_tokenizer"] = False
        if use_server:
            model_args["base_url"] = f"{server_endpoint}/v1/completions"
            model_args["tokenized_requests"] = False
        logger.info(f"Running lm-eval on tasks {eval_args.lm_eval_tasks}")
        is_ernie = "ernie" in model_name.lower()
        logger.warning(f"Is Ernie: {is_ernie}, using batch size 1")
        if use_server:
            results = evaluator.simple_evaluate(
                model="local-completions",
                model_args=model_args,
                tasks=eval_args.lm_eval_tasks,
                num_fewshot=0,
                random_seed=seed,
                numpy_random_seed=seed,
                torch_random_seed=seed,
                batch_size=eval_args.parallel_tasks if not is_ernie else 1,
                apply_chat_template=False,
                fewshot_as_multiturn=False,
            )
        else:
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=model_args,
                tasks=eval_args.lm_eval_tasks,
                num_fewshot=0,
                batch_size="auto",
                random_seed=seed,
                numpy_random_seed=seed,
                torch_random_seed=seed,
                apply_chat_template=False,
                fewshot_as_multiturn=False,
            )
        try:
            with open(f"{results_file_base_name}_table.txt", "w") as f:
                print(make_table(results))
                print(make_table(results), file=f)
                if "groups" in results:
                    print(make_table(results, "groups"))
                    print(make_table(results, "groups"), file=f)
            with open(f"{results_file_base_name}.json", "w") as f:
                json.dump(results, f)
        except Exception as e:
            pass
        logger.info(f"Finished evaluating lm-eval")

    try:
        if eval_args.run_evalplus:
            enable_thinking = True
            if "qwen" in model_name.lower() or "glm-4.5" in model_name.lower():
                logger.info("Disabling thinking for Qwen/GLM models")
                enable_thinking = False
            for task in eval_args.evalplus_tasks:
                logger.info(f"Running evalplus on task {task}")
                output_file = results_dir / f"{task}.json"
                # evalplus fork
                if use_server:
                    evalplus_evaluator(
                        model=model_name,
                        root=results_dir / "evalplus_results",
                        dataset=task,
                        backend="openai",
                        attn_implementation="flash_attention_2",
                        greedy=eval_args.greedy,
                        output_file=output_file,
                        base_url=f"{server_endpoint}/v1",
                        temperature=eval_args.temperature
                        if not eval_args.greedy
                        else 0.0,
                        enable_thinking=enable_thinking,
                        parallel_tasks=eval_args.parallel_tasks,
                    )
                else:
                    evalplus_evaluator(
                        model=model_name,
                        root=results_dir / "evalplus_results",
                        dataset=task,
                        backend="hf",
                        attn_implementation="flash_attention_2",
                        greedy=eval_args.greedy,
                        output_file=output_file,
                        temperature=eval_args.temperature
                        if not eval_args.greedy
                        else 0.0,
                        enable_thinking=enable_thinking,
                    )
    except Exception as e:
        logger.error(f"An error occurred during evalplus: {e}")
        raise e
        pass
    try:
        if eval_args.run_livecodebench:
            if not use_server:
                raise ValueError(
                    "Current LCB ReapBase model style implementation requries a vLLM server to be running"
                )
            from lcb_runner.runner.main import main as lcb_main
            from lcb_runner.runner.main import get_args_dict

            original_model, uncompressed_model = get_original_model_name(model_name)

            lcb_args = get_args_dict(
                model=original_model,
                n=1,
                output_path=results_dir,
                enable_thinking=False,
                base_url=f"{server_endpoint}/v1",
                start_date="2025-01-01",
                end_date="2025-07-31",
                evaluate=True,
                timeout=120,
                local_model_path=model_name if not uncompressed_model else None,
                max_tokens=16384,
            )
            logger.info(f"Running LiveCodeBench with args: {lcb_args}")
            lcb_main(lcb_args)
            logger.info(f"Finished evaluating LiveCodeBench")
    except Exception as e:
        logger.error(f"An error occurred during livecodebench: {e}")
        pass
    try:
        if eval_args.run_wildbench:
            from helm.benchmark.run import helm_run, create_helm_run_args
            from helm.common.hierarchical_logger import setup_default_logging

            original_model, uncompressed_model = get_original_model_name(model_name)
            judge_model_path = os.environ.get("WILDBENCH_JUDGE_MODEL_PATH")
            if not judge_model_path:
                raise ValueError(
                    "WILDBENCH_JUDGE_MODEL_PATH must point to a local judge checkpoint."
                )
            judge_model_name = os.environ.get(
                "WILDBENCH_JUDGE_MODEL_NAME", "google/gemma-4-E2B-it"
            )
            judge_port = int(
                os.environ.get("WILDBENCH_JUDGE_PORT", str(eval_args.vllm_port + 1000))
            )
            judge_base_url = f"http://127.0.0.1:{judge_port}"
            judge_log = results_dir / f"wildbench_judge_{judge_port}.log"
            judge_env = os.environ.copy()
            judge_env["CUDA_VISIBLE_DEVICES"] = os.environ.get(
                "WILDBENCH_JUDGE_CUDA_VISIBLE_DEVICES", "0"
            )
            judge_command = [
                "vllm",
                "serve",
                judge_model_path,
                "--port",
                str(judge_port),
                "--served-model-name",
                judge_model_name,
                "--dtype",
                "bfloat16",
                "--gpu-memory-utilization",
                os.environ.get("WILDBENCH_JUDGE_GPU_MEM_UTIL", "0.15"),
                "--max-model-len",
                os.environ.get("WILDBENCH_JUDGE_MAX_MODEL_LEN", "16384"),
                "--max-num-seqs",
                os.environ.get("WILDBENCH_JUDGE_MAX_NUM_SEQS", "8"),
                "--trust-remote-code",
                "--disable-log-requests",
            ]
            logger.info("Starting WildBench judge: %s", " ".join(judge_command))
            judge_log_handle = open(judge_log, "w")
            judge_process = subprocess.Popen(
                judge_command,
                stdout=judge_log_handle,
                stderr=subprocess.STDOUT,
                env=judge_env,
            )
            wait_for_server(judge_base_url)
            start_server_watchdog(
                judge_process,
                base_url=judge_base_url,
                server_log_path=str(judge_log),
            )

            local_path = results_dir / f"wildbench_prod_env_{eval_args.vllm_port}"
            local_path.mkdir(parents=True, exist_ok=True)
            candidate_deployment = "vllm/wildbench-candidate"
            judge_deployment = "vllm/wildbench-gemma-judge"
            with (local_path / "model_metadata.yaml").open("w") as f:
                yaml.safe_dump(
                    {
                        "models": [
                            {
                                "name": original_model,
                                "display_name": "REAP-conditioned merged Qwen3",
                                "description": "Local REAP-conditioned merged model",
                                "creator_organization_name": "Qwen",
                                "access": "open",
                                "release_date": "2025-07-29",
                                "tags": ["TEXT_MODEL_TAG", "INSTRUCTION_FOLLOWING_MODEL_TAG"],
                            },
                            {
                                "name": judge_model_name,
                                "display_name": "Gemma WildBench judge",
                                "description": "Local Gemma judge",
                                "creator_organization_name": "Google",
                                "access": "open",
                                "release_date": "2026-01-01",
                                "tags": ["TEXT_MODEL_TAG", "INSTRUCTION_FOLLOWING_MODEL_TAG"],
                            },
                        ]
                    },
                    f,
                    sort_keys=False,
                )
            with (local_path / "model_deployments.yaml").open("w") as f:
                yaml.safe_dump(
                    {
                        "model_deployments": [
                            {
                                "name": candidate_deployment,
                                "model_name": original_model,
                                "tokenizer_name": original_model,
                                "max_sequence_length": 32768,
                                "client_spec": {
                                    "class_name": "helm.clients.vllm_client.VLLMChatClient",
                                    "args": {
                                        "base_url": f"{server_endpoint}/v1",
                                        "vllm_model_name": original_model,
                                    },
                                },
                            },
                            {
                                "name": judge_deployment,
                                "model_name": judge_model_name,
                                "tokenizer_name": judge_model_path,
                                "max_sequence_length": 16384,
                                "client_spec": {
                                    "class_name": "helm.clients.vllm_client.VLLMChatClient",
                                    "args": {
                                        "base_url": f"{judge_base_url}/v1",
                                        "vllm_model_name": judge_model_name,
                                    },
                                },
                            },
                        ]
                    },
                    f,
                    sort_keys=False,
                )
            os.environ["WILDBENCH_JUDGE_NAME"] = "gemma"
            os.environ["WILDBENCH_JUDGE_MODEL_NAME"] = judge_model_name
            os.environ["WILDBENCH_JUDGE_DEPLOYMENT"] = judge_deployment

            suite = "test"
            run_entries = [f"wildbench:subset=v2,model={original_model}"]
            helm_args = create_helm_run_args(
                suite=suite,
                local_path=local_path,
                run_entries=run_entries,
                output_path=f"{results_dir}/wildbench",
                max_eval_instances=(
                    int(os.environ["WILDBENCH_MAX_EVAL_INSTANCES"])
                    if os.environ.get("WILDBENCH_MAX_EVAL_INSTANCES")
                    else None
                ),
                cache_instances=True,
                disable_cache=False,
            )
            logger.info(f"Running WildBench with args: {helm_args}")
            setup_default_logging()
            helm_run(helm_args)
            logger.info(f"Finished evaluating WildBench")
    except Exception as e:
        logger.error(f"An error occurred during wildbench: {e}")
        pass
    finally:
        if "judge_process" in locals():
            judge_process.terminate()
        if "judge_log_handle" in locals():
            judge_log_handle.close()
    if eval_args.run_math:
        try:
            from evalscope.run import run_task, TaskConfig

            # Which math datasets to run (comma-separated). The offline modelscope
            # gsm8k loader can hang (~200s/item); set MATH_DATASETS=math_500 to run
            # math500-only. Defaults to both for backward compatibility.
            math_datasets = [
                d.strip()
                for d in os.environ.get("MATH_DATASETS", "gsm8k,math_500").split(",")
                if d.strip()
            ]
            task_config = TaskConfig(
                model=model_name,
                generation_config={
                    "do_sample": False,
                    "max_new_tokens": 16384,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                datasets=math_datasets,
                api_url=f"{server_endpoint}/v1",
                api_key="EMPTY",
                timeout=3600,
                work_dir=results_dir / "evalscope_results",
                dataset_args={
                    "gsm8k": {
                        "few_shot_num": 0,
                    }
                },
                eval_batch_size=int(os.environ.get("VLLM_MATH_BATCH", "8")),
                eval_type="service",
            )
            logger.info(f"Running evalscope math with config: {task_config}")
            run_task(task_config)
            logger.info(f"Finished evaluating evalscope math benchmarks")
        except Exception as e:
            logger.error(f"An error occurred during math evaluation: {e}")
            pass

    if use_server:
        process.terminate()
    if use_server and "process" in locals():
        process.terminate()


if __name__ == "__main__":
    parser = HfArgumentParser((ReapArgs, ModelArgs, EvalArgs))
    reap_args, model_args, eval_args = parser.parse_args_into_dataclasses()
    run_evaluate(
        model_args=model_args,
        results_dir=eval_args.results_dir,
        eval_args=eval_args,
        seed=reap_args.seed,
    )
