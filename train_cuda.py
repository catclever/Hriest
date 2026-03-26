import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import os
import sys
# Auto-resolve the parent workspace root and force it to the FRONT of the import path
# This prevents the local `model.py` file from aggressively shadowing the parent `model/` package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

from training.core.dataloader import MultiEmbDataLoader
from training.core.char_tokenizer import CharTokenizer
from training.core.checkpoint import Checkpointer
from model.config import ModelConfig
from model_cuda import TinyCharEncoderCUDA, GodEncoderCUDA, SensoryFuserCUDA, load_mlx_safetensors_into_torch

def custom_lr_schedule(global_step: int, max_lr: float, warmup_steps: int):
    # Absolute linear warmup from 0
    if global_step < warmup_steps:
        return max_lr * (global_step / warmup_steps)
    return max_lr

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
        
    config = ModelConfig()
    config.max_seq_len = args.max_seq_len
    
    tokenizer = CharTokenizer()
    d_model = config.decoder_heads * 64
    
    # 1. Instantiate the Frozen PyTorch Teacher Replicas
    fuser = SensoryFuserCUDA(config.emb_dims, d_model).to(device)
    god_encoder = GodEncoderCUDA(d_model, config.z_dim).to(device)
    
    print(f"\n[Bridging MLX -> PyTorch] Loading Frozen Teacher Safetensors from {args.p0_ckpt}...")
    fuser_path = f"{args.p0_ckpt}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{args.p0_ckpt}/sense_adapter.safetensors"
        
    load_mlx_safetensors_into_torch(fuser, fuser_path)
    load_mlx_safetensors_into_torch(god_encoder, f"{args.p0_ckpt}/god_encoder.safetensors")
    
    # 2. Instantiate the Tiny Student Encoder
    print("[Student] Initializing PyTorch TinyCharEncoder (FlashAttention-2 + RoPE enabled)...")
    student = TinyCharEncoderCUDA(
        vocab_size=config.vocab_size,
        d_model=1024,
        n_heads=8,
        n_layers=6,
        max_seq_len=config.max_seq_len,
        z_dim=config.z_dim
    ).to(device)
    
    # 3. Memory-Mapped Generic Array Dataloader using PyTorch Tensor Backend injection
    emb_files = [
        "data/Basic_ZH/embs/hy-tmp/roberta_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/gte_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/bge_embeddings.npy",
        "data/Basic_ZH/embs/hy-tmp/text2vec_embeddings.npy"
    ]
    
    dataloader = MultiEmbDataLoader(
        parquet_path="data/Basic_ZH/chunked_mixed_wiki.parquet",
        emb_paths=emb_files,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        backend='torch' # Highly critical flag dynamically translating memory maps to torch.Tensors
    )
    
    # 4. Universal Checkpointer
    checkpointer = Checkpointer(args.out_dir, prefix=args.ckpt_prefix)
    checkpointer.register_model("student", student)
    checkpointer.register_dataloader("dataloader", dataloader)
    checkpointer.register_args(args)
    
    start_step = checkpointer.load_latest()
    
    # 5. Optimizer & AMP Scaler
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Enable Automatic Mixed Precision for NVIDIA architectures. MPS lacks autocast support.
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # 6. Training Loop
    global_step = start_step
    
    print(f"\n========================================================")
    print(f"🚀 PyTorch Main Engine Ignited! Target Epochs: {args.epochs}")
    print(f"⚙️  Batch: {args.batch_size} | Configured AMP: {use_amp} | Peak LR: {args.lr}")
    print(f"========================================================\n")
    
    try:
        for epoch in range(dataloader.current_epoch, args.epochs):
            for token_inputs, batch_embs, attention_mask in dataloader:
                global_step += 1
                
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
                     
    except KeyboardInterrupt:
        print(f"\n[Interrupt] Caught PyTorch kill signal at step {global_step}! Emergency save triggered.")
        checkpointer.save(global_step, is_emergency=True)

if __name__ == "__main__":
    main()
