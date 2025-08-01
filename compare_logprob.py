import dataclasses
import gc
import math
import requests
import json
from argparse import Namespace
from statistics import mean, median, stdev
from typing import Any

import torch
import torch.nn.functional as F
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

PROMPTS = [
    "One of the most important things in life is to",
    "The answer to 1 + 1 is",
]

# (model, logprobs_mode) -> (vllm_errs, vllm_prob_errs, sglang_errs, sglang_prob_errs)
global_results: dict[tuple[str, str], Any] = {}

stat_name_to_func = {
    "max": max,
    "mean": mean,
    "stdev": stdev,
    "median": median,
    "min": min
}


def get_hf_outputs(args: EngineArgs) -> tuple[list[list[int]], torch.Tensor]:
    """Get HuggingFace model outputs for comparison."""
    model_config = args.create_model_config()
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_config.dtype,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    hf_model.eval()
    hf_input_ids = []
    all_logprobs = []
    
    with torch.no_grad():
        for prompt in PROMPTS:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            hf_input_ids.append(input_ids[0].tolist())
            
            outputs = hf_model(input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            logprobs = F.log_softmax(logits, dim=-1)
            all_logprobs.append(logprobs)
    
    # Stack all logprobs
    max_len = max(lp.shape[0] for lp in all_logprobs)
    padded_logprobs = []
    for lp in all_logprobs:
        if lp.shape[0] < max_len:
            padding = torch.zeros(max_len - lp.shape[0], lp.shape[1], 
                                device=lp.device, dtype=lp.dtype)
            lp = torch.cat([lp, padding], dim=0)
        padded_logprobs.append(lp)
    
    hf_logprobs = torch.stack(padded_logprobs)  # [batch, seq_len, vocab_size]
    
    # Clean up
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return hf_input_ids, hf_logprobs


def get_vllm_outputs(
    args: EngineArgs,
    sampling_params: SamplingParams,
) -> list[dict[str, Any]]:
    """Get vLLM outputs."""
    llm = LLM(**dataclasses.asdict(args))
    outputs = llm.generate(
        PROMPTS,
        sampling_params=sampling_params,
        )
    final_outputs = []
    for output in outputs:
        final_outputs.append({
            "input_ids": output.prompt_token_ids,
            "output_ids": output.outputs[0].token_ids,
            "logprobs": output.outputs[0].logprobs,
        })
    # Release memory for next benchmark
    del llm
    gc.collect()
    return final_outputs


def get_sglang_outputs(
    model_path: str,
    sampling_params: SamplingParams,
    sglang_url: str = "http://localhost:2379",
    use_offline: bool = False
) -> list[dict[str, Any]]:
    """Get SGLang outputs via HTTP API or offline mode."""
    
    if use_offline:
        return get_sglang_outputs_offline(model_path, sampling_params)
    
    # Prepare request for SGLang
    sg_req = {
        "prompt": PROMPTS,
        "max_tokens": 0,  # Only get prompt logprobs
        "temperature": sampling_params.temperature,
        "logprobs": 1,  # Enable logprobs
        "echo": True,   # Include prompt in response
    }
    
    try:
        response = requests.post(
            f"{sglang_url}/v1/completions",
            json=sg_req,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        sg_json = response.json()
        
        final_outputs = []
        for i, choice in enumerate(sg_json["choices"]):
            # Extract token IDs and logprobs
            token_logprobs = choice["logprobs"]["token_logprobs"]
            tokens = choice["logprobs"]["tokens"]
            
            # Convert tokens to IDs if needed
            # Note: SGLang might return token strings, need to convert to IDs
            token_ids = []
            logprobs_dict_list = []
            
            for j, (token, logprob_data) in enumerate(zip(tokens, token_logprobs)):
                if isinstance(logprob_data, dict):
                    token_id = logprob_data.get("token_id", None)
                    logprob = logprob_data.get("logprob", None)
                else:
                    # Handle different SGLang response format
                    token_id = j  # Fallback
                    logprob = logprob_data
                
                token_ids.append(token_id)
                logprobs_dict_list.append({token_id: logprob} if token_id is not None else {})
            
            final_outputs.append({
                "input_ids": token_ids,  # Prompt token IDs
                "output_ids": [],  # No generation, only prompt
                "logprobs": logprobs_dict_list,
            })
            
        return final_outputs
        
    except requests.RequestException as e:
        print(f"Error connecting to SGLang server: {e}")
        print("Make sure SGLang server is running at", sglang_url)
        return []
    except Exception as e:
        print(f"Error processing SGLang response: {e}")
        return []


def get_sglang_outputs_offline(
    model_path: str,
    sampling_params: SamplingParams,
) -> list[dict[str, Any]]:
    """Get SGLang outputs using offline mode (direct model loading)."""
    try:
        import sglang as sgl
        from sglang import Engine
        import torch   
        print(f"Loading SGLang model from {model_path}...")
        
        # Initialize engine
        engine = Engine(
             model_path=model_path,
             trust_remote_code=True,
             dtype="float16"
         )

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        final_outputs = []

        for prompt in PROMPTS:
            # Tokenize the prompt first
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            
            # Run SGLang generate with logprob enabled
            try:
                # Try SGLang's SamplingParams if available
                from sglang import SamplingParams as SGSamplingParams
                sg_sampling_params = SGSamplingParams(
                    temperature=sampling_params.temperature,
                    top_p=1.0,
                    top_k=-1,
                    max_new_tokens=1,
                    return_logprob=True
                )
                outputs = engine.generate([prompt], sampling_params=sg_sampling_params)
            except:
                # Fallback: try basic generate call
                outputs = engine.generate([prompt])
            
            # Extract logprobs from SGLang output
            logprobs_list = []
            if hasattr(outputs[0], 'meta_info') and 'prompt_logprobs' in outputs[0].meta_info:
                prompt_logprobs = outputs[0].meta_info['prompt_logprobs']
                for pos, token_id in enumerate(input_ids):
                    if pos < len(prompt_logprobs):
                        logprob = prompt_logprobs[pos].get(token_id, 0.0)
                        logprobs_list.append({token_id: logprob})
                    else:
                        logprobs_list.append({token_id: 0.0})
            else:
                # Fallback: use first token position = 0
                for token_id in input_ids:
                    logprobs_list.append({token_id: 0.0})
            
            final_outputs.append({
                "input_ids": input_ids,
                "output_ids": [],
                "logprobs": logprobs_list,
            })
        
        # Clean up
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        
        return final_outputs
        
    except ImportError:
        print("SGLang not installed. Please install with: pip install sglang")
        return []
    except Exception as e:
        print(f"Error in SGLang offline mode: {e}")
        return []


def compare_logprob_three_way(args: EngineArgs, sampling_params: SamplingParams, sglang_url: str = "http://localhost:30000", use_sglang_offline: bool = False):
    """Compare logprobs between HuggingFace, vLLM, and SGLang."""
    
    print("Getting HuggingFace outputs...")
    hf_input_ids, hf_logprobs = get_hf_outputs(args)
    
    # Prepare HF records
    hf_records = []
    for i, prompt_ids in enumerate(hf_input_ids):
        for pos, tok_id in enumerate(prompt_ids):
            lp = hf_logprobs[i, pos-1, tok_id].item() if pos > 0 else 0.0
            hf_records.append((i, pos, tok_id, lp))
    
    print("Getting vLLM outputs...")
    v_outs = get_vllm_outputs(args, sampling_params)
    v_records = []
    for i, rec in enumerate(v_outs):
        for pos, logp_dict in enumerate(rec["logprobs"]):
            if logp_dict and rec["input_ids"]:  # Check if we have valid data
                tok = rec["input_ids"][pos] if pos < len(rec["input_ids"]) else None
                if tok is not None:
                    lp = logp_dict.get(int(tok), None)
                    v_records.append((i, pos, int(tok), lp))
    
    print("Getting SGLang outputs...")
    if use_sglang_offline:
        print("Using SGLang offline mode...")
    else:
        print("Using SGLang HTTP API mode...")
    sg_outs = get_sglang_outputs(args.model, sampling_params, sglang_url, use_sglang_offline)
    sg_records = []
    for i, rec in enumerate(sg_outs):
        for pos, logp_dict in enumerate(rec["logprobs"]):
            if logp_dict and rec["input_ids"]:  # Check if we have valid data
                tok = rec["input_ids"][pos] if pos < len(rec["input_ids"]) else None
                if tok is not None:
                    lp = list(logp_dict.values())[0] if logp_dict else None
                    sg_records.append((i, pos, int(tok), lp))
    
    # Align and compare results
    print(f"HF records: {len(hf_records)}, vLLM records: {len(v_records)}, SGLang records: {len(sg_records)}")
    
    diffs = []
    min_len = min(len(hf_records), len(v_records), len(sg_records))
    
    for idx in range(min_len):
        hf = hf_records[idx]
        v = v_records[idx] if idx < len(v_records) else (hf[0], hf[1], hf[2], None)
        sg = sg_records[idx] if idx < len(sg_records) else (hf[0], hf[1], hf[2], None)
        
        # Check alignment
        if hf[2] != v[2] or hf[2] != sg[2]:
            print(f"Token mismatch at position {idx}: HF={hf[2]}, vLLM={v[2]}, SGLang={sg[2]}")
            continue
            
        diff_entry = {
            "prompt_idx": hf[0],
            "pos": hf[1],
            "token_id": hf[2],
            "hf_lp": hf[3],
            "vllm_lp": v[3],
            "sglang_lp": sg[3],
            "diff_vllm_hf": v[3] - hf[3] if v[3] is not None else None,
            "diff_sglang_hf": sg[3] - hf[3] if sg[3] is not None else None,
            "diff_sglang_vllm": sg[3] - v[3] if sg[3] is not None and v[3] is not None else None,
        }
        diffs.append(diff_entry)
    
    # Analyze differences
    print("\n=== Comparison Results ===")
    
    if diffs:
        # vLLM vs HF
        vllm_hf_diffs = [d["diff_vllm_hf"] for d in diffs if d["diff_vllm_hf"] is not None]
        if vllm_hf_diffs:
            print(f"vLLM vs HF differences - Mean: {mean(vllm_hf_diffs):.6f}, "
                  f"Max: {max(abs(d) for d in vllm_hf_diffs):.6f}, "
                  f"Std: {stdev(vllm_hf_diffs) if len(vllm_hf_diffs) > 1 else 0:.6f}")
        
        # SGLang vs HF
        sglang_hf_diffs = [d["diff_sglang_hf"] for d in diffs if d["diff_sglang_hf"] is not None]
        if sglang_hf_diffs:
            print(f"SGLang vs HF differences - Mean: {mean(sglang_hf_diffs):.6f}, "
                  f"Max: {max(abs(d) for d in sglang_hf_diffs):.6f}, "
                  f"Std: {stdev(sglang_hf_diffs) if len(sglang_hf_diffs) > 1 else 0:.6f}")
        
        # SGLang vs vLLM
        sglang_vllm_diffs = [d["diff_sglang_vllm"] for d in diffs if d["diff_sglang_vllm"] is not None]
        if sglang_vllm_diffs:
            print(f"SGLang vs vLLM differences - Mean: {mean(sglang_vllm_diffs):.6f}, "
                  f"Max: {max(abs(d) for d in sglang_vllm_diffs):.6f}, "
                  f"Std: {stdev(sglang_vllm_diffs) if len(sglang_vllm_diffs) > 1 else 0:.6f}")
        
        # Show first few detailed comparisons
        print("\n=== Sample Detailed Comparisons ===")
        headers = ["Prompt", "Pos", "Token", "HF LogP", "vLLM LogP", "SGLang LogP", "vLLM-HF", "SGLang-HF", "SGLang-vLLM"]
        table_data = []
        for i, d in enumerate(diffs[:10]):  # Show first 10
            table_data.append([
                d["prompt_idx"],
                d["pos"],
                d["token_id"],
                f"{d['hf_lp']:.4f}" if d['hf_lp'] is not None else "None",
                f"{d['vllm_lp']:.4f}" if d['vllm_lp'] is not None else "None",
                f"{d['sglang_lp']:.4f}" if d['sglang_lp'] is not None else "None",
                f"{d['diff_vllm_hf']:.4f}" if d['diff_vllm_hf'] is not None else "None",
                f"{d['diff_sglang_hf']:.4f}" if d['diff_sglang_hf'] is not None else "None",
                f"{d['diff_sglang_vllm']:.4f}" if d['diff_sglang_vllm'] is not None else "None",
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    return diffs


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Compare logprobs between HF, vLLM, and SGLang")
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument("--sglang-url", type=str, default="http://localhost:30000",
                       help="SGLang server URL")
    parser.add_argument("--sglang-offline", action="store_true",
                       help="Use SGLang offline mode (direct model loading) instead of HTTP API")
    args = parser.parse_args()
    
    engine_args = EngineArgs.from_cli_args(args)
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=1,     # Only get prompt logprobs
        logprobs=1,       # Enable logprobs
        prompt_logprobs=1
    )
    
    print(f"Comparing logprobs for model: {engine_args.model}")
    if args.sglang_offline:
        print("Using SGLang offline mode")
    else:
        print(f"SGLang URL: {args.sglang_url}")
    
    diffs = compare_logprob_three_way(engine_args, sampling_params, args.sglang_url, args.sglang_offline)
    
    if diffs:
        print(f"\nProcessed {len(diffs)} token comparisons successfully.")
    else:
        print("\nNo comparisons could be made. Check your setup.")
