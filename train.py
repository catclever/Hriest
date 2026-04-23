import argparse
import os
import re
import json
import glob
import numpy as np
import shutil
import threading
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder
from training.core.dataloader import MultiEmbDataLoader, ChunkedNpzDataLoader
from training.char_tokenizer import CharTokenizer
from training.core.checkpoint import Checkpointer
from training.core.schedule import linear_warmup_schedule
from distilled_emb.model import TinyCharEncoder

def get_distill_parser():
    parser = argparse.ArgumentParser(description="Standalone Embedding Distillation (Teacher -> Student)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4) # Slightly higher for small student
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--out_dir", type=str, default="checkpoints/distilled")
    parser.add_argument("--ckpt_prefix", type=str, default="tinybert_v1")
    parser.add_argument("--p0_ckpt", type=str, required=True, help="Frozen Phase 0 weights to distill from")
    parser.add_argument("--config_file", type=str, default=None, help="Optional dataset config (e.g. HollyShit dataset_config.json)")
    parser.add_argument("--emb_paths", nargs="*", default=None, help="Explicit embedding .npy paths for MultiEmbDataLoader")
    parser.add_argument("--emb_dir", type=str, default="data/Basic_ZH/embs/hy-tmp", help="Auto-discovery dir for *embeddings*.npy")
    parser.add_argument("--parquet_path", type=str, default=None, help="Parquet path override; falls back to config or legacy default")
    parser.add_argument("--data_dir", type=str, default="./embs", help="Local read directory for chunked npz files")
    parser.add_argument("--download_cache_dir", type=str, default=None, help="Temporary download cache dir for chunked npz; defaults to --data_dir")
    parser.add_argument("--chunk_size", type=int, default=500000, help="Chunk size used by ChunkedNpzDataLoader")
    parser.add_argument("--strict_route_check", action=argparse.BooleanOptionalAction, default=True, help="Require checkpoint routes and data routes to match exactly")
    parser.add_argument("--student_d_model", type=int, default=1024, help="Student hidden size")
    parser.add_argument("--student_n_heads", type=int, default=8, help="Student attention heads")
    parser.add_argument("--student_n_layers", type=int, default=6, help="Student transformer layers")
    return parser


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
    from safetensors import safe_open

    fuser_path = os.path.join(p0_ckpt, "sense_fuser.safetensors")
    if not os.path.exists(fuser_path):
        fuser_path = os.path.join(p0_ckpt, "sense_adapter.safetensors")
    if not os.path.exists(fuser_path):
        raise FileNotFoundError(f"Missing sense_fuser/sense_adapter checkpoint under: {p0_ckpt}")

    god_path = os.path.join(p0_ckpt, "god_encoder.safetensors")
    if not os.path.exists(god_path):
        raise FileNotFoundError(f"Missing god_encoder.safetensors under: {p0_ckpt}")

    route_dims = {}
    d_model = None
    with safe_open(fuser_path, framework="np") as sf:
        for key in sf.keys():
            m = re.match(r"adapters\.(\d+)\.(?:net\.layers\.0|fc1)\.weight$", key)
            if m:
                idx = int(m.group(1))
                tensor = sf.get_tensor(key)
                route_dims[idx] = int(tensor.shape[1])
                d_model = int(tensor.shape[0])

    if not route_dims:
        raise RuntimeError(f"Could not infer adapter routes from {fuser_path}")
    emb_dims = [route_dims[i] for i in sorted(route_dims.keys())]

    z_dim = None
    with safe_open(god_path, framework="np") as sf:
        for key in sf.keys():
            if key in {"net.layers.2.weight", "fc2.weight"}:
                tensor = sf.get_tensor(key)
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
            backend='mlx'
        )

    emb_paths = resolve_emb_paths(args, ds_config, expected_routes)
    print(f"[Data] Using MultiEmbDataLoader ({len(emb_paths)} routes): {emb_paths}")
    return MultiEmbDataLoader(
        parquet_path=parquet_path,
        emb_paths=emb_paths,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        backend='mlx'
    )


def attach_chunk_download_router(loader, local_data_dir: str, download_cache_dir: str):
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
        nonlocal _last_temp_cleanup_at
        if now_ts is None:
            now_ts = time.time()
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
                    arr = np.load(path)["features"]
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
        os.path.exists(os.path.join(ckpt_dir, "student.safetensors"))
        or os.path.exists(os.path.join(ckpt_dir, "student.pt"))
    )


def _is_checkpoint_dir_name(folder_name: str):
    if folder_name == "latest_emergency" or folder_name.startswith("step_"):
        return True
    if folder_name.endswith("_latest_emergency"):
        return True
    return bool(re.match(r".+_step_\d+$", folder_name))


def is_resume_checkpoint_valid(ckpt_dir: str):
    if not has_student_weights(ckpt_dir):
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

    opt_path = os.path.join(ckpt_dir, "optimizer.safetensors")
    if not os.path.exists(opt_path):
        return False, "missing optimizer.safetensors"
    try:
        mx.load(opt_path)
    except Exception as e:
        return False, f"broken optimizer.safetensors ({e})"

    return True, ""


def find_latest_checkpoint_across_all(out_dir: str, preferred_prefix: str = None):
    if not os.path.isdir(out_dir):
        return None, preferred_prefix or ""

    candidates = []
    for d in os.listdir(out_dir):
        full = os.path.join(out_dir, d)
        if not os.path.isdir(full):
            continue
        if not has_student_weights(full):
            continue
        step = infer_checkpoint_step(full, d)
        prefix = extract_prefix_from_checkpoint_name(d)
        mtime = os.path.getmtime(full)
        prefix_rank = 1 if (preferred_prefix is not None and prefix == preferred_prefix) else 0
        candidates.append((step, prefix_rank, mtime, full, prefix, d))

    if not candidates:
        return None, preferred_prefix or ""

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    for _, _, _, ckpt_dir, prefix, folder_name in candidates:
        ok, reason = is_resume_checkpoint_valid(ckpt_dir)
        if ok:
            return ckpt_dir, prefix
        print(f"[Resume] Dropping invalid checkpoint: {ckpt_dir} | reason: {reason}")
        if _is_checkpoint_dir_name(folder_name):
            try:
                shutil.rmtree(ckpt_dir)
                print(f"[Resume] Removed broken checkpoint folder: {ckpt_dir}")
            except Exception as e:
                print(f"[Resume] Failed to remove broken checkpoint folder: {ckpt_dir} ({e})")
    return None, preferred_prefix or ""

def main():
    parser = get_distill_parser()
    args = parser.parse_args()

    ds_config = load_optional_config(args.config_file)
    if args.download_cache_dir is None:
        args.download_cache_dir = args.data_dir

    config = ModelConfig()
    config.max_seq_len = args.max_seq_len

    tokenizer = CharTokenizer()

    fuser_path, god_path, emb_dims, d_model, z_dim = infer_teacher_arch_from_ckpt(args.p0_ckpt)
    print(
        f"[Teacher Arch] routes={len(emb_dims)} emb_dims={emb_dims} "
        f"d_model={d_model} z_dim={z_dim}"
    )

    dataloader = build_dataloader(args, tokenizer, ds_config, expected_routes=len(emb_dims))
    dataloader = attach_chunk_download_router(dataloader, args.data_dir, args.download_cache_dir)

    # 1. Load the frozen Super-Teacher Target (GodEncoder)
    fuser = SensoryFuser(emb_dims, d_model)
    god_encoder = GodEncoder(d_model, z_dim)

    print(f"Loading Frozen Teacher weights from {args.p0_ckpt}...")
    fuser.load_weights(fuser_path)
    god_encoder.load_weights(god_path)

    fuser.freeze()
    god_encoder.freeze()
    print("Teacher modules frozen.")

    # 2. Instantiate the Tiny Student Encoder
    print("Initializing Student NativeCharEncoder (6 layers, 1024 dimensions)...")
    student = TinyCharEncoder(
        vocab_size=config.vocab_size,
        d_model=args.student_d_model,
        n_heads=args.student_n_heads,
        n_layers=args.student_n_layers,
        max_seq_len=config.max_seq_len,
        z_dim=z_dim
    )
    mx.eval(student.parameters())

    # 3. Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr)

    # 4. Checkpointer
    checkpointer = Checkpointer(args.out_dir, prefix=args.ckpt_prefix)
    checkpointer.register_model("student", student)
    checkpointer.register_optimizer("optimizer", optimizer)
    checkpointer.register_dataloader("dataloader", dataloader)
    checkpointer.register_args(args)

    resume_dir, resume_prefix = find_latest_checkpoint_across_all(
        args.out_dir, preferred_prefix=args.ckpt_prefix
    )
    if resume_dir:
        checkpointer.prefix = resume_prefix
        print(f"[Resume] Auto-selected latest checkpoint across all: {resume_dir}")
        start_step = checkpointer.load(resume_dir)
    else:
        print(f"[Resume] No valid checkpoint found under {args.out_dir}. Starting from step 0.")
        start_step = 0

    # 5. Loss Definition (MSE)
    def loss_fn(model, token_inputs, attention_mask, z_target_truth):
        z_pred = model(token_inputs, attention_mask)
        # Pure MSE against GodEncoder's space perfectly aligns direction and magnitude
        loss = mx.mean(mx.square(z_pred - z_target_truth))
        return loss

    loss_and_grad_fn = nn.value_and_grad(student, loss_fn)

    @mx.compile
    def step_fn(token_inputs, attention_mask, z_target_truth):
        loss, grads = loss_and_grad_fn(student, token_inputs, attention_mask, z_target_truth)
        clipped_grads, _ = optim.clip_grad_norm(grads, 1.0)
        return loss, clipped_grads

    # 6. Training Loop
    global_step = start_step

    print(f"\n========================================================")
    print(f"Starting Distillation Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"Target Pure Semantic Space: {args.p0_ckpt}")
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

                # 8a. Get Absolute Truth from GodEncoder (Frozen, computed on the fly via RAM)
                # Ensure None weights correctly hit Centroid Fusion
                f_t = fuser(batch_embs, weights=None)
                z_target_truth = god_encoder(f_t) # Shape: (B, 1, z_dim) or (B, z_dim)

                if len(z_target_truth.shape) == 3:
                     z_target_truth = mx.squeeze(z_target_truth, axis=1)

                # Sync LR natively
                optimizer.learning_rate = linear_warmup_schedule(global_step, args.lr, args.warmup_steps)

                # 8b. Train Student (JIT Compiled Fwd+Bwd, Python Opt Update)
                loss, grads = step_fn(token_inputs, attention_mask, z_target_truth)
                optimizer.update(student, grads)
                mx.eval(student.parameters(), optimizer.state, loss)

                if global_step % 10 == 0:
                     print(f"Epoch {epoch+1} | Step {global_step} | Distill MSE: {loss.item():.4f}")

                if global_step % args.save_steps == 0:
                     checkpointer.save(global_step)

    except KeyboardInterrupt:
        print(f"\n[Interrupt] Caught kill signal at step {global_step}! Emergency save triggered.")
        checkpointer.save(global_step, is_emergency=True)

if __name__ == "__main__":
    main()
