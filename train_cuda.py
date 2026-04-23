import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
import re
import json
import glob
import shutil
import threading
import time
import zipfile
# Auto-resolve the parent workspace root and force it to the FRONT of the import path
# This prevents the local `model.py` file from aggressively shadowing the parent `model/` package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    # Move script dir to the end so package-style `model.*` can be resolved first.
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]
import argparse

from training.core.dataloader import MultiEmbDataLoader, ChunkedNpzDataLoader
from training.core.char_tokenizer import CharTokenizer
from training.core.checkpoint import Checkpointer
from model.config import ModelConfig
from model_cuda import TinyCharEncoderCUDA, GodEncoderCUDA, SensoryFuserCUDA, load_mlx_safetensors_into_torch

def custom_lr_schedule(global_step: int, max_lr: float, warmup_steps: int):
    # Absolute linear warmup from 0
    if global_step < warmup_steps:
        return max_lr * (global_step / warmup_steps)
    return max_lr


def load_optional_config(config_file: str):
    if not config_file:
        return {}
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path_maybe_relative(path_value: str, base_dir: str):
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    candidate = os.path.join(base_dir, path_value)
    if os.path.exists(candidate):
        return candidate
    return path_value


def infer_teacher_arch_from_ckpt(p0_ckpt: str):
    from safetensors.torch import load_file

    fuser_path = os.path.join(p0_ckpt, "sense_fuser.safetensors")
    if not os.path.exists(fuser_path):
        fuser_path = os.path.join(p0_ckpt, "sense_adapter.safetensors")
    if not os.path.exists(fuser_path):
        raise FileNotFoundError(f"Missing sense_fuser/sense_adapter checkpoint under: {p0_ckpt}")

    god_path = os.path.join(p0_ckpt, "god_encoder.safetensors")
    if not os.path.exists(god_path):
        raise FileNotFoundError(f"Missing god_encoder.safetensors under: {p0_ckpt}")

    fuser_state = load_file(fuser_path)
    god_state = load_file(god_path)

    route_dims = {}
    d_model = None
    for key, tensor in fuser_state.items():
        m = re.match(r"adapters\.(\d+)\.(?:net\.layers\.0|fc1)\.weight$", key)
        if m:
            idx = int(m.group(1))
            route_dims[idx] = int(tensor.shape[1])
            d_model = int(tensor.shape[0])

    if not route_dims:
        raise RuntimeError(f"Could not infer adapter routes from {fuser_path}")

    emb_dims = [route_dims[i] for i in sorted(route_dims.keys())]

    z_dim = None
    for key, tensor in god_state.items():
        if key in {"net.layers.2.weight", "fc2.weight"}:
            z_dim = int(tensor.shape[0])
            if d_model is None:
                d_model = int(tensor.shape[1])
            break

    if z_dim is None:
        raise RuntimeError(f"Could not infer z_dim from {god_path}")
    if d_model is None:
        raise RuntimeError(f"Could not infer d_model from {fuser_path}/{god_path}")

    return fuser_path, god_path, emb_dims, d_model, z_dim


def resolve_emb_paths(args, ds_config: dict, expected_routes: int):
    if args.emb_paths:
        emb_paths = args.emb_paths
    elif ds_config.get("emb_paths"):
        emb_paths = ds_config["emb_paths"]
    else:
        emb_paths = sorted(glob.glob(os.path.join(args.emb_dir, "*embeddings*.npy")))

    if len(emb_paths) != expected_routes:
        msg = (
            f"Embedding route mismatch: expected {expected_routes} from checkpoint, "
            f"but got {len(emb_paths)} embedding files. paths={emb_paths}"
        )
        if args.strict_route_check:
            raise ValueError(msg)
        if len(emb_paths) < expected_routes:
            raise ValueError(msg + " | Non-strict cannot recover from fewer routes.")
        print(f"[Warn] {msg} | Non-strict mode: truncating to first {expected_routes} paths.")
        emb_paths = emb_paths[:expected_routes]
    return emb_paths


def build_dataloader(args, tokenizer, ds_config: dict, expected_routes: int):
    has_chunked_conf = bool(ds_config.get("base_models") and ds_config.get("chunk_name_patterns"))
    config_base_dir = os.path.dirname(os.path.abspath(args.config_file)) if args.config_file else os.getcwd()
    parquet_path = args.parquet_path or ds_config.get("parquet_path", "data/Basic_ZH/chunked_mixed_wiki.parquet")
    parquet_path = resolve_path_maybe_relative(parquet_path, config_base_dir)

    if has_chunked_conf:
        models = ds_config["base_models"]
        chunk_patterns = ds_config["chunk_name_patterns"]
        ms_repo_id = ds_config.get("modelscope_repo_id")
        if len(models) != expected_routes:
            msg = (
                f"Model route mismatch: checkpoint expects {expected_routes}, "
                f"but config has {len(models)} base_models={models}"
            )
            if args.strict_route_check:
                raise ValueError(msg)
            if len(models) < expected_routes:
                raise ValueError(msg + " | Non-strict cannot recover from fewer routes.")
            print(f"[Warn] {msg} | Non-strict mode: truncating to first {expected_routes} models.")
            models = models[:expected_routes]
            chunk_patterns = {k: v for k, v in chunk_patterns.items() if k in models}
        print(f"[Data] Using ChunkedNpzDataLoader ({len(models)} routes, dynamic ModelScope/local chunks).")
        return ChunkedNpzDataLoader(
            parquet_path=parquet_path,
            models=models,
            chunk_patterns=chunk_patterns,
            tokenizer=tokenizer,
            ms_repo_id=ms_repo_id,
            chunk_size=args.chunk_size,
            local_npz_dir=args.data_dir,
            cache_dir=args.download_cache_dir,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            shuffle=True,
            lazy_start=True,
            backend='torch'
        )

    emb_paths = resolve_emb_paths(args, ds_config, expected_routes)
    print(f"[Data] Using MultiEmbDataLoader ({len(emb_paths)} routes): {emb_paths}")
    return MultiEmbDataLoader(
        parquet_path=parquet_path,
        emb_paths=emb_paths,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        backend='torch'
    )


def attach_chunk_download_router(loader, local_data_dir: str, download_cache_dir: str):
    # Only patch ChunkedNpzDataLoader-like objects.
    if not hasattr(loader, "models") or not hasattr(loader, "chunk_patterns"):
        return loader

    local_data_dir = os.path.abspath(local_data_dir)
    download_cache_dir = os.path.abspath(download_cache_dir)
    loader.npz_dir = local_data_dir
    loader.cache_dir = download_cache_dir
    _cand_locks = {}
    _cand_locks_guard = threading.Lock()
    _last_temp_cleanup_at = 0.0

    def _cleanup_stale_temp_artifacts(now_ts: float = None, stale_seconds: int = 600):
        """
        Best-effort cleanup for leftover temp artifacts from interrupted downloads.
        We only delete stale files to avoid racing with active downloads.
        """
        nonlocal _last_temp_cleanup_at
        if now_ts is None:
            now_ts = time.time()
        # Avoid scanning too frequently.
        if now_ts - _last_temp_cleanup_at < 30:
            return
        _last_temp_cleanup_at = now_ts

        roots = {
            os.path.join(local_data_dir, "._____temp"),
            os.path.join(download_cache_dir, "._____temp"),
            local_data_dir,
            download_cache_dir,
        }
        deleted_files = 0
        deleted_dirs = 0
        for root in roots:
            if not root or not os.path.exists(root):
                continue
            for dp, _, fs in os.walk(root, topdown=False):
                for fn in fs:
                    fp = os.path.join(dp, fn)
                    try:
                        age = now_ts - os.path.getmtime(fp)
                    except OSError:
                        continue
                    if age < stale_seconds:
                        continue
                    # Temp naming patterns observed in HF/ModelScope caches.
                    is_temp = (
                        fn.endswith(".incomplete")
                        or fn.endswith(".tmp")
                        or fn.endswith(".partial")
                        or ".tmp." in fn
                    )
                    if is_temp:
                        try:
                            os.remove(fp)
                            deleted_files += 1
                        except OSError:
                            pass
                # Remove empty leaf dirs under temp roots.
                try:
                    if dp.startswith(os.path.join(local_data_dir, "._____temp")) or dp.startswith(
                        os.path.join(download_cache_dir, "._____temp")
                    ):
                        if not os.listdir(dp):
                            os.rmdir(dp)
                            deleted_dirs += 1
                except OSError:
                    pass
        if deleted_files or deleted_dirs:
            print(f"[DataLoader] 🧹 清理临时残留: files={deleted_files}, dirs={deleted_dirs}")

    def _get_cand_lock(cand_name: str):
        with _cand_locks_guard:
            if cand_name not in _cand_locks:
                _cand_locks[cand_name] = threading.Lock()
            return _cand_locks[cand_name]

    if local_data_dir == download_cache_dir:
        print(f"[Data] Download+read directory unified at: {local_data_dir}")
    else:
        print(
            f"[Data] Download cache: {download_cache_dir} | Read directory: {local_data_dir} "
            f"(download then stage into read dir)"
        )

    def _resolve_npz_path(cand_name: str):
        _cleanup_stale_temp_artifacts()
        local_path = os.path.join(local_data_dir, cand_name)
        cand_lock = _get_cand_lock(cand_name)
        with cand_lock:
            if os.path.exists(local_path):
                return local_path

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if loader.ms_repo_id:
                from modelscope.hub.file_download import dataset_file_download
                last_err = None
                downloaded_path = None
                for retry in range(3):
                    try:
                        downloaded_path = dataset_file_download(
                            loader.ms_repo_id, cand_name, cache_dir=download_cache_dir
                        )
                        break
                    except FileNotFoundError as e:
                        # ModelScope temp-file race under concurrent access; retry is safe.
                        last_err = e
                        time.sleep(0.5)
                if downloaded_path is None:
                    raise last_err
                downloaded_path = os.path.abspath(downloaded_path)
            else:
                downloaded_path = os.path.join(local_data_dir, cand_name)
                if not os.path.exists(downloaded_path):
                    raise FileNotFoundError(f"本地目标块缺失：{downloaded_path}")

            if downloaded_path != os.path.abspath(local_path):
                try:
                    os.replace(downloaded_path, local_path)
                except OSError:
                    shutil.copy2(downloaded_path, local_path)
                    try:
                        os.remove(downloaded_path)
                    except OSError:
                        pass
            return local_path

    loader._resolve_npz_path = _resolve_npz_path

    # Keep async prefetch behavior, but make it always read from local_data_dir.
    def _prefetch_worker_routed(start_macro_ptr):
        worker_macro_ptr = start_macro_ptr
        while not loader.stop_event.is_set():
            if worker_macro_ptr >= len(loader.macro_chunk_indices):
                loader.prefetch_queue.put(None)
                return

            chunk_idx = loader.macro_chunk_indices[worker_macro_ptr]
            start_idx, end_idx = loader.chunk_bounds[chunk_idx]

            try:
                active_chunk_embs = []
                cached_paths = []
                for model_name in loader.models:
                    pattern = loader.chunk_patterns.get(model_name, f"{model_name}_chunk_{{start:07d}}_{{end:07d}}.npz")
                    model_end = loader._per_model_bounds.get(model_name, {}).get(start_idx, end_idx)
                    cand_name = pattern.format(start=start_idx, end=model_end)

                    path = loader._resolve_npz_path(cand_name)
                    arr = load_npz_features_with_repair(loader, path, cand_name=cand_name)
                    active_chunk_embs.append(arr)
                    cached_paths.append(path)

                cur_chunk_size = end_idx - start_idx
                active_micro_indices = np.arange(cur_chunk_size)
                if loader.shuffle:
                    rng = np.random.default_rng(loader.seed + worker_macro_ptr + loader.current_epoch)
                    rng.shuffle(active_micro_indices)

                payload = {
                    "embs": active_chunk_embs,
                    "micro_indices": active_micro_indices,
                    "global_start": start_idx,
                    "macro_idx_ptr": worker_macro_ptr,
                    "cached_paths": cached_paths,
                }
                loader.prefetch_queue.put(payload)
                worker_macro_ptr += 1
            except Exception as e:
                print(f"\n[DataLoader] 后台打工仔抓取/加载失败 (网络可能断开了？): {e}")
                print(f"[DataLoader] 这不是致命错误！打工仔将在 10 秒后原地重新尝试拉取 Chunk {chunk_idx}...")
                import time
                time.sleep(10)
                continue

    loader._prefetch_worker = _prefetch_worker_routed
    return loader


def load_npz_features_with_repair(loader, path: str, cand_name: str = None, max_retries: int = 2):
    attempt = 0
    last_err = None
    while attempt <= max_retries:
        try:
            return np.load(path)["features"]
        except (zipfile.BadZipFile, OSError, ValueError) as e:
            last_err = e
            attempt += 1
            if attempt > max_retries:
                break
            print(f"[Warn] Corrupted npz detected: {path} ({e}). retry {attempt}/{max_retries}")
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

            if cand_name and hasattr(loader, "_resolve_npz_path"):
                path = loader._resolve_npz_path(cand_name)
            elif cand_name and loader.ms_repo_id:
                from modelscope.hub.file_download import dataset_file_download
                path = dataset_file_download(loader.ms_repo_id, cand_name, cache_dir=loader.cache_dir)
            else:
                raise
    raise RuntimeError(f"Failed to load npz after retries: {path} | last_error={last_err}")


def disable_chunk_prefetch(loader):
    # Only patch ChunkedNpzDataLoader-like objects.
    if not hasattr(loader, "chunk_bounds") or not hasattr(loader, "macro_chunk_indices"):
        return loader

    # Stop any existing worker thread if it has been started.
    try:
        if hasattr(loader, "stop_event"):
            loader.stop_event.set()
        if hasattr(loader, "prefetch_thread") and loader.prefetch_thread.is_alive():
            loader.prefetch_thread.join(timeout=1.0)
    except Exception:
        pass

    def _load_payload_sync(macro_ptr):
        if macro_ptr >= len(loader.macro_chunk_indices):
            return None

        chunk_idx = loader.macro_chunk_indices[macro_ptr]
        start_idx, end_idx = loader.chunk_bounds[chunk_idx]
        active_chunk_embs = []
        cached_paths = []

        for model_name in loader.models:
            pattern = loader.chunk_patterns.get(
                model_name, f"{model_name}_chunk_{{start:07d}}_{{end:07d}}.npz"
            )
            model_end = loader._per_model_bounds.get(model_name, {}).get(start_idx, end_idx)
            cand_name = pattern.format(start=start_idx, end=model_end)

            if hasattr(loader, "_resolve_npz_path"):
                path = loader._resolve_npz_path(cand_name)
            elif loader.ms_repo_id:
                from modelscope.hub.file_download import dataset_file_download
                path = dataset_file_download(loader.ms_repo_id, cand_name, cache_dir=loader.cache_dir)
            else:
                path = os.path.join(loader.npz_dir, cand_name)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"本地目标块缺失：{path}")

            arr = load_npz_features_with_repair(loader, path, cand_name=cand_name)
            active_chunk_embs.append(arr)
            cached_paths.append(path)

        cur_chunk_size = end_idx - start_idx
        active_micro_indices = np.arange(cur_chunk_size)
        if loader.shuffle:
            rng = np.random.default_rng(loader.seed + macro_ptr + loader.current_epoch)
            rng.shuffle(active_micro_indices)

        return {
            "embs": active_chunk_embs,
            "micro_indices": active_micro_indices,
            "global_start": start_idx,
            "macro_idx_ptr": macro_ptr,
            "cached_paths": cached_paths,
        }

    def _start_prefetching_sync(start_macro_ptr):
        loader._sync_next_macro_ptr = start_macro_ptr

    def _pop_next_chunk_sync():
        if loader.auto_cleanup and getattr(loader, "_prev_chunk_paths", []):
            loader._cleanup_cached_files(loader._prev_chunk_paths)
            loader._prev_chunk_paths = []

        if not hasattr(loader, "_sync_next_macro_ptr"):
            if getattr(loader, "active_chunk_embs", None) is None:
                loader._sync_next_macro_ptr = getattr(loader, "active_macro_idx_ptr", 0)
            else:
                loader._sync_next_macro_ptr = getattr(loader, "active_macro_idx_ptr", 0) + 1

        payload = _load_payload_sync(loader._sync_next_macro_ptr)
        if payload is None:
            raise StopIteration

        loader.active_chunk_embs = payload["embs"]
        loader.active_micro_indices = payload["micro_indices"]
        loader.active_global_start = payload["global_start"]
        loader.active_macro_idx_ptr = payload["macro_idx_ptr"]
        loader.active_micro_ptr = 0
        loader._prev_chunk_paths = payload.get("cached_paths", [])
        loader._sync_next_macro_ptr += 1

        if not hasattr(loader, "emb_dims"):
            loader.emb_dims = [arr.shape[-1] for arr in loader.active_chunk_embs]
            print(f"[DataLoader] 自动侦测到多路特征维度: {loader.emb_dims}")

        print(f"\n[DataLoader] 🐢 同步按需加载新块 (全局起点: {loader.active_global_start})")

    loader._start_prefetching = _start_prefetching_sync
    loader._pop_next_chunk = _pop_next_chunk_sync
    if getattr(loader, "active_chunk_embs", None) is None:
        loader._started = False
    else:
        loader._sync_next_macro_ptr = getattr(loader, "active_macro_idx_ptr", 0) + 1
    return loader


def enable_download_only_prefetch(loader):
    # Download chunks ahead in background, but load npz arrays only on demand.
    if not hasattr(loader, "chunk_bounds") or not hasattr(loader, "macro_chunk_indices"):
        return loader

    # Stop any existing prefetch worker first.
    try:
        if hasattr(loader, "stop_event"):
            loader.stop_event.set()
        if hasattr(loader, "prefetch_thread") and loader.prefetch_thread.is_alive():
            loader.prefetch_thread.join(timeout=1.0)
    except Exception:
        pass

    def _download_chunk_only(macro_ptr):
        if macro_ptr >= len(loader.macro_chunk_indices):
            return False
        chunk_idx = loader.macro_chunk_indices[macro_ptr]
        start_idx, end_idx = loader.chunk_bounds[chunk_idx]
        for model_name in loader.models:
            pattern = loader.chunk_patterns.get(
                model_name, f"{model_name}_chunk_{{start:07d}}_{{end:07d}}.npz"
            )
            model_end = loader._per_model_bounds.get(model_name, {}).get(start_idx, end_idx)
            cand_name = pattern.format(start=start_idx, end=model_end)
            if hasattr(loader, "_resolve_npz_path"):
                loader._resolve_npz_path(cand_name)
            elif loader.ms_repo_id:
                from modelscope.hub.file_download import dataset_file_download
                dataset_file_download(loader.ms_repo_id, cand_name, cache_dir=loader.cache_dir)
            else:
                path = os.path.join(loader.npz_dir, cand_name)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"本地目标块缺失：{path}")
        return True

    def _download_worker(start_macro_ptr, stop_event):
        macro_ptr = start_macro_ptr
        while not stop_event.is_set():
            if macro_ptr >= len(loader.macro_chunk_indices):
                return
            chunk_idx = loader.macro_chunk_indices[macro_ptr]
            try:
                _download_chunk_only(macro_ptr)
                macro_ptr += 1
            except Exception as e:
                print(f"\n[DataLoader] 后台下载失败: {e}")
                print(f"[DataLoader] 10 秒后重试 Chunk {chunk_idx}（仅下载模式）...")
                time.sleep(10)

    def _start_prefetching_download_only(start_macro_ptr):
        # Stop previous background downloader.
        if hasattr(loader, "_dl_stop_event") and loader._dl_stop_event is not None:
            loader._dl_stop_event.set()
        if hasattr(loader, "_dl_thread") and loader._dl_thread is not None and loader._dl_thread.is_alive():
            loader._dl_thread.join(timeout=1.0)

        loader._dl_stop_event = threading.Event()
        loader._dl_thread = threading.Thread(
            target=_download_worker, args=(start_macro_ptr, loader._dl_stop_event), daemon=True
        )
        loader._dl_thread.start()
        loader._sync_next_macro_ptr = start_macro_ptr

    def _pop_next_chunk_sync():
        if loader.auto_cleanup and getattr(loader, "_prev_chunk_paths", []):
            loader._cleanup_cached_files(loader._prev_chunk_paths)
            loader._prev_chunk_paths = []

        if not hasattr(loader, "_sync_next_macro_ptr"):
            if getattr(loader, "active_chunk_embs", None) is None:
                loader._sync_next_macro_ptr = getattr(loader, "active_macro_idx_ptr", 0)
            else:
                loader._sync_next_macro_ptr = getattr(loader, "active_macro_idx_ptr", 0) + 1

        macro_ptr = loader._sync_next_macro_ptr
        if macro_ptr >= len(loader.macro_chunk_indices):
            raise StopIteration

        chunk_idx = loader.macro_chunk_indices[macro_ptr]
        start_idx, end_idx = loader.chunk_bounds[chunk_idx]
        active_chunk_embs = []
        cached_paths = []

        for model_name in loader.models:
            pattern = loader.chunk_patterns.get(
                model_name, f"{model_name}_chunk_{{start:07d}}_{{end:07d}}.npz"
            )
            model_end = loader._per_model_bounds.get(model_name, {}).get(start_idx, end_idx)
            cand_name = pattern.format(start=start_idx, end=model_end)

            if hasattr(loader, "_resolve_npz_path"):
                path = loader._resolve_npz_path(cand_name)
            elif loader.ms_repo_id:
                from modelscope.hub.file_download import dataset_file_download
                path = dataset_file_download(loader.ms_repo_id, cand_name, cache_dir=loader.cache_dir)
            else:
                path = os.path.join(loader.npz_dir, cand_name)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"本地目标块缺失：{path}")

            arr = load_npz_features_with_repair(loader, path, cand_name=cand_name)
            active_chunk_embs.append(arr)
            cached_paths.append(path)

        cur_chunk_size = end_idx - start_idx
        active_micro_indices = np.arange(cur_chunk_size)
        if loader.shuffle:
            rng = np.random.default_rng(loader.seed + macro_ptr + loader.current_epoch)
            rng.shuffle(active_micro_indices)

        loader.active_chunk_embs = active_chunk_embs
        loader.active_micro_indices = active_micro_indices
        loader.active_global_start = start_idx
        loader.active_macro_idx_ptr = macro_ptr
        loader.active_micro_ptr = 0
        loader._prev_chunk_paths = cached_paths
        loader._sync_next_macro_ptr += 1

        if not hasattr(loader, "emb_dims"):
            loader.emb_dims = [arr.shape[-1] for arr in loader.active_chunk_embs]
            print(f"[DataLoader] 自动侦测到多路特征维度: {loader.emb_dims}")

        print(f"\n[DataLoader] 📦 仅下载预取模式：同步装载块 (全局起点: {loader.active_global_start})")

    loader._start_prefetching = _start_prefetching_download_only
    loader._pop_next_chunk = _pop_next_chunk_sync
    if getattr(loader, "active_chunk_embs", None) is None:
        loader._started = False
    else:
        loader._sync_next_macro_ptr = getattr(loader, "active_macro_idx_ptr", 0) + 1
    return loader


def expected_checkpoint_dir(out_dir: str, prefix: str, step: int, is_emergency: bool = False):
    base_name = "latest_emergency" if is_emergency else f"step_{step}"
    folder_name = f"{prefix}_{base_name}" if prefix else base_name
    return os.path.join(out_dir, folder_name)


def extract_prefix_from_checkpoint_name(folder_name: str):
    if folder_name == "latest_emergency" or folder_name.startswith("step_"):
        return ""
    if folder_name.endswith("_latest_emergency"):
        return folder_name[: -len("_latest_emergency")]
    m = re.match(r"(.+)_step_\d+$", folder_name)
    if m:
        return m.group(1)
    return ""


def infer_checkpoint_step(ckpt_dir: str, folder_name: str):
    state_file = os.path.join(ckpt_dir, "dataloader.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            return int(state.get("global_step", -1))
        except Exception:
            pass
    m = re.search(r"_step_(\d+)$", folder_name)
    if m:
        return int(m.group(1))
    m = re.match(r"step_(\d+)$", folder_name)
    if m:
        return int(m.group(1))
    return -1


def has_student_weights(ckpt_dir: str):
    return (
        os.path.exists(os.path.join(ckpt_dir, "student.pt"))
        or os.path.exists(os.path.join(ckpt_dir, "student.safetensors"))
    )


def _is_checkpoint_dir_name(folder_name: str):
    if folder_name == "latest_emergency" or folder_name.startswith("step_"):
        return True
    if folder_name.endswith("_latest_emergency"):
        return True
    return bool(re.match(r".+_step_\d+$", folder_name))


def is_resume_checkpoint_valid(ckpt_dir: str):
    """
    A checkpoint is resumable only when critical files exist and are readable.
    """
    student_ok = has_student_weights(ckpt_dir)
    if not student_ok:
        return False, "missing student weights"

    state_file = os.path.join(ckpt_dir, "dataloader.json")
    if not os.path.exists(state_file):
        return False, "missing dataloader.json"
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        _ = int(state.get("global_step", -1))
    except Exception as e:
        return False, f"broken dataloader.json ({e})"

    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    if not os.path.exists(opt_path):
        return False, "missing optimizer.pt"
    try:
        torch.load(opt_path, map_location="cpu")
    except Exception as e:
        return False, f"broken optimizer.pt ({e})"

    return True, ""


def find_latest_checkpoint_across_all(out_dir: str, preferred_prefix: str = None):
    if not os.path.isdir(out_dir):
        return None, 0, preferred_prefix or ""

    candidates = []
    for d in os.listdir(out_dir):
        full = os.path.join(out_dir, d)
        if not os.path.isdir(full):
            continue
        # Only treat true training checkpoints as resume candidates.
        if not has_student_weights(full):
            continue
        step = infer_checkpoint_step(full, d)
        prefix = extract_prefix_from_checkpoint_name(d)
        mtime = os.path.getmtime(full)
        prefix_rank = 1 if (preferred_prefix is not None and prefix == preferred_prefix) else 0
        candidates.append((step, prefix_rank, mtime, full, prefix, d))

    if not candidates:
        return None, 0, preferred_prefix or ""

    # Highest global_step wins; if tied, prefer preferred_prefix; if still tied, newer mtime.
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    for step, _, _, ckpt_dir, prefix, folder_name in candidates:
        ok, reason = is_resume_checkpoint_valid(ckpt_dir)
        if ok:
            return ckpt_dir, max(step, 0), prefix
        print(f"[Resume] Dropping invalid checkpoint: {ckpt_dir} | reason: {reason}")
        # Auto-remove obviously broken checkpoint folders so next run won't pick them again.
        if _is_checkpoint_dir_name(folder_name):
            try:
                shutil.rmtree(ckpt_dir)
                print(f"[Resume] Removed broken checkpoint folder: {ckpt_dir}")
            except Exception as e:
                print(f"[Resume] Failed to remove broken checkpoint folder: {ckpt_dir} ({e})")
    return None, 0, preferred_prefix or ""


def find_checkpoint_dir_by_step(out_dir: str, prefix: str, step: int):
    if not os.path.isdir(out_dir):
        return None
    candidates = []
    for d in os.listdir(out_dir):
        full = os.path.join(out_dir, d)
        if not os.path.isdir(full):
            continue
        if prefix:
            if not (d.startswith(f"{prefix}_step_") or d == f"{prefix}_latest_emergency"):
                continue
        elif not (d.startswith("step_") or d == "latest_emergency"):
            continue
        state_file = os.path.join(full, "dataloader.json")
        if not os.path.exists(state_file):
            continue
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            if int(state.get("global_step", -1)) == int(step):
                candidates.append(full)
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_torch_optimizer_state(optimizer, out_dir: str, prefix: str, step: int):
    if step <= 0:
        return False
    ckpt_dir = find_checkpoint_dir_by_step(out_dir, prefix, step)
    if not ckpt_dir:
        return False
    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    if not os.path.exists(opt_path):
        return False
    state = torch.load(opt_path, map_location="cpu")
    optimizer.load_state_dict(state)
    print(f"[Resume] Loaded optimizer state from: {opt_path}")
    return True


def load_torch_optimizer_state_from_dir(optimizer, ckpt_dir: str):
    if not ckpt_dir:
        return False
    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    if not os.path.exists(opt_path):
        return False
    state = torch.load(opt_path, map_location="cpu")
    optimizer.load_state_dict(state)
    print(f"[Resume] Loaded optimizer state from: {opt_path}")
    return True


def save_torch_optimizer_state(optimizer, out_dir: str, prefix: str, step: int, is_emergency: bool = False):
    ckpt_dir = expected_checkpoint_dir(out_dir, prefix, step, is_emergency=is_emergency)
    os.makedirs(ckpt_dir, exist_ok=True)
    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    torch.save(optimizer.state_dict(), opt_path)
    print(f"[Checkpoint] Saved optimizer state to {opt_path}")


def get_distill_parser():
    parser = argparse.ArgumentParser(description="PyTorch Standalone Embedding Distillation (Teacher -> Student)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4) # Safe default LR for deep networks
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--out_dir", type=str, default="checkpoints/distilled")
    parser.add_argument("--ckpt_prefix", type=str, default="tinybert_pt_v1")
    parser.add_argument("--p0_ckpt", type=str, required=True, help="Frozen Phase 0 MLX weights to distill from")
    parser.add_argument("--config_file", type=str, default=None, help="Optional dataset config (e.g. HollyShit dataset_config.json)")
    parser.add_argument("--emb_paths", nargs="*", default=None, help="Explicit embedding .npy paths for MultiEmbDataLoader")
    parser.add_argument("--emb_dir", type=str, default="data/Basic_ZH/embs/hy-tmp", help="Auto-discovery dir for *embeddings*.npy")
    parser.add_argument("--parquet_path", type=str, default=None, help="Parquet path override; falls back to config or legacy default")
    parser.add_argument("--data_dir", type=str, default="./embs", help="Local read directory for chunked npz files")
    parser.add_argument("--download_cache_dir", type=str, default=None, help="Temporary download cache dir for chunked npz; defaults to --data_dir")
    parser.add_argument("--chunk_size", type=int, default=500000, help="Chunk size used by ChunkedNpzDataLoader")
    parser.add_argument("--strict_route_check", action=argparse.BooleanOptionalAction, default=True, help="Require checkpoint routes and data routes to match exactly")
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True, help="Enable async chunk prefetch for ChunkedNpzDataLoader")
    parser.add_argument("--download_only_prefetch", action="store_true", help="Background download ahead but do not preload npz arrays into memory")
    parser.add_argument("--student_d_model", type=int, default=1024, help="Student hidden size")
    parser.add_argument("--student_n_heads", type=int, default=8, help="Student attention heads")
    parser.add_argument("--student_n_layers", type=int, default=6, help="Student transformer layers")
    return parser

def main():
    parser = get_distill_parser()
    args = parser.parse_args()
    
    # Dual-Backend Hardward Detector
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\n🔥 [Hardware] CUDA cluster detected. Booting NVIDIA PyTorch Amp Backend...")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n🍏 [Hardware] Apple Silicon detected. Booting PyTorch MPS Accelerated Backend...")
    else:
        device = torch.device("cpu")
        print("\n🐢 [Hardware] CPU only detected.")
        
    ds_config = load_optional_config(args.config_file)
    if args.download_cache_dir is None:
        args.download_cache_dir = args.data_dir

    config = ModelConfig()
    config.max_seq_len = args.max_seq_len

    tokenizer = CharTokenizer()

    fuser_path, god_path, emb_dims, d_model, z_dim = infer_teacher_arch_from_ckpt(args.p0_ckpt)
    print(
        f"\n[Teacher Arch] routes={len(emb_dims)} emb_dims={emb_dims} "
        f"d_model={d_model} z_dim={z_dim}"
    )

    dataloader = build_dataloader(args, tokenizer, ds_config, expected_routes=len(emb_dims))
    dataloader = attach_chunk_download_router(dataloader, args.data_dir, args.download_cache_dir)
    if args.download_only_prefetch:
        dataloader = enable_download_only_prefetch(dataloader)
        print("[Data] Download-only prefetch enabled: async download ahead, sync npz load on demand.")
    elif not args.prefetch:
        dataloader = disable_chunk_prefetch(dataloader)
        print("[Data] Async prefetch disabled: running in synchronous on-demand chunk loading mode.")

    # 1. Instantiate the Frozen PyTorch Teacher Replicas
    fuser = SensoryFuserCUDA(emb_dims, d_model).to(device)
    god_encoder = GodEncoderCUDA(d_model, z_dim).to(device)

    print(f"\n[Bridging MLX -> PyTorch] Loading Frozen Teacher Safetensors from {args.p0_ckpt}...")
    load_mlx_safetensors_into_torch(fuser, fuser_path)
    load_mlx_safetensors_into_torch(god_encoder, god_path)

    # 2. Instantiate the Tiny Student Encoder
    print("[Student] Initializing PyTorch TinyCharEncoder (FlashAttention-2 + RoPE enabled)...")
    student = TinyCharEncoderCUDA(
        vocab_size=config.vocab_size,
        d_model=args.student_d_model,
        n_heads=args.student_n_heads,
        n_layers=args.student_n_layers,
        max_seq_len=config.max_seq_len,
        z_dim=z_dim
    ).to(device)

    # 3. Optimizer & AMP Scaler
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 4. Universal Checkpointer
    checkpointer = Checkpointer(args.out_dir, prefix=args.ckpt_prefix)
    checkpointer.register_model("student", student)
    checkpointer.register_dataloader("dataloader", dataloader)
    checkpointer.register_args(args)

    resume_dir, _, resume_prefix = find_latest_checkpoint_across_all(
        args.out_dir, preferred_prefix=args.ckpt_prefix
    )
    if resume_dir:
        checkpointer.prefix = resume_prefix
        print(f"[Resume] Auto-selected latest checkpoint across all: {resume_dir}")
        start_step = checkpointer.load(resume_dir)
        if not load_torch_optimizer_state_from_dir(optimizer, resume_dir):
            # Fallback keeps backward compatibility with older folders.
            load_torch_optimizer_state(optimizer, args.out_dir, checkpointer.prefix, start_step)
    else:
        print(f"[Resume] No checkpoint found under {args.out_dir}. Starting from step 0.")
        start_step = 0

    # 5. Training Loop
    global_step = start_step
    
    print(f"\n========================================================")
    print(f"🚀 PyTorch Main Engine Ignited! Target Epochs: {args.epochs}")
    print(f"⚙️  Batch: {args.batch_size} | Configured AMP: {use_amp} | Peak LR: {args.lr}")
    print(f"========================================================\n")
    
    try:
        for epoch in range(dataloader.current_epoch, args.epochs):
            for token_inputs, batch_embs, attention_mask in dataloader:
                global_step += 1
                if len(batch_embs) != len(emb_dims):
                    msg = (
                        f"Runtime embedding route mismatch: got {len(batch_embs)} routes, "
                        f"expected {len(emb_dims)} from teacher checkpoint"
                    )
                    if args.strict_route_check:
                        raise ValueError(msg)
                    if len(batch_embs) < len(emb_dims):
                        raise ValueError(msg + " | Non-strict cannot recover from fewer routes.")
                    print(f"[Warn] {msg} | Non-strict mode: truncating runtime routes.")
                    batch_embs = batch_embs[:len(emb_dims)]
                
                # A: Manual LR Schedule injection
                lr = custom_lr_schedule(global_step, args.lr, args.warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # B: Async Non-Blocking Tensor Memory Transfer
                token_inputs = token_inputs.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                batch_embs = [emb.to(device, non_blocking=True) for emb in batch_embs]
                
                # C: Zero the Grads
                optimizer.zero_grad(set_to_none=True)
                
                # D: Forward Pass (Using Dynamic AMP casting)
                # Note: No custom compilation closure necessary in PyTorch natively
                with torch.amp.autocast('cuda', enabled=use_amp):
                    # Ground Truth Target Logic
                    with torch.no_grad():
                        f_t = fuser(batch_embs, weights=None) 
                        z_target_truth = god_encoder(f_t)
                        if len(z_target_truth.shape) == 3:
                             z_target_truth = z_target_truth.squeeze(1)
                             
                    # Student Logic (The heavy lifting graph)
                    z_pred = student(token_inputs, attention_mask)
                    # MSE Loss against the exact Absolute Spatial Dimensional Vector
                    loss = F.mse_loss(z_pred, z_target_truth)
                    
                # E: Backward execution with AMP Loss Scaling
                scaler.scale(loss).backward()
                
                # F: Crucial Global Infinity Prevention Gradient Clipping (Airbag Mechanism)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                
                # G: Opt Execute
                scaler.step(optimizer)
                scaler.update()
                
                if global_step % 10 == 0:
                     print(f"Epoch {epoch+1} | Step {global_step} | Distill PyTorch MSE: {loss.item():.4f}")
                     
                if global_step % args.save_steps == 0:
                     checkpointer.save(global_step)
                     save_torch_optimizer_state(optimizer, args.out_dir, checkpointer.prefix, global_step)
                     
    except KeyboardInterrupt:
        print(f"\n[Interrupt] Caught PyTorch kill signal at step {global_step}! Emergency save triggered.")
        checkpointer.save(global_step, is_emergency=True)
        save_torch_optimizer_state(optimizer, args.out_dir, checkpointer.prefix, global_step, is_emergency=True)

if __name__ == "__main__":
    main()
