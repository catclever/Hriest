import argparse
import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder
from training.core.dataloader import MultiEmbDataLoader
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
    return parser

def main():
    parser = get_distill_parser()
    args = parser.parse_args()
    
    config = ModelConfig()
    config.max_seq_len = args.max_seq_len
    
    tokenizer = CharTokenizer()
    d_model = config.decoder_heads * 64
    
    # 1. Load the frozen Super-Teacher Target (GodEncoder)
    fuser = SensoryFuser(config.emb_dims, d_model)
    god_encoder = GodEncoder(d_model, config.z_dim)
    
    print(f"Loading Frozen Teacher weights from {args.p0_ckpt}...")
    fuser_path = f"{args.p0_ckpt}/sense_fuser.safetensors"
    if not os.path.exists(fuser_path):
        fuser_path = f"{args.p0_ckpt}/sense_adapter.safetensors"
        
    fuser.load_weights(fuser_path)
    god_encoder.load_weights(f"{args.p0_ckpt}/god_encoder.safetensors")
    
    fuser.freeze()
    god_encoder.freeze()
    print("Teacher modules frozen.")
    
    # 2. Instantiate the Tiny Student Encoder
    print("Initializing Student NativeCharEncoder (6 layers, 1024 dimensions)...")
    student = TinyCharEncoder(
        vocab_size=config.vocab_size,
        d_model=1024,
        n_heads=8,
        n_layers=6,
        max_seq_len=config.max_seq_len,
        z_dim=config.z_dim
    )
    mx.eval(student.parameters())
    
    # 3. Dataloader (Phase 0 DataLoader is perfect because we want single chunks, not contiguous Phase 1 trajectories)
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
        max_seq_len=args.max_seq_len
    )
    
    # 4. Checkpointer
    checkpointer = Checkpointer(args.out_dir, prefix=args.ckpt_prefix)
    checkpointer.register_model("student", student)
    checkpointer.register_dataloader("dataloader", dataloader)
    checkpointer.register_args(args)
    
    start_step = checkpointer.load_latest()
    # 5. Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr)
    
    # 6. Loss Definition (MSE)
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
    
    # 7. Training Loop
    global_step = start_step
    
    print(f"\n========================================================")
    print(f"Starting Distillation Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"Target Pure Semantic Space: {args.p0_ckpt}")
    print(f"========================================================\n")
    
    try:
        for epoch in range(dataloader.current_epoch, args.epochs):
            for token_inputs, batch_embs, attention_mask in dataloader:
                global_step += 1
                
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
