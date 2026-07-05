import os
import sys
import argparse
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from distilled_emb.model_cuda import TinyCharEncoderCUDA, WeakDecoderCUDA, load_mlx_safetensors_into_torch
from training.core.char_tokenizer import CharTokenizer
from model.config import WeakDecoderConfig
import json
import glob

def get_latest_student_ckpt(distilled_dir):
    ckpts = glob.glob(os.path.join(distilled_dir, "*step_*"))
    if not ckpts:
        return None
    # Sort by modification time
    return max(ckpts, key=os.path.getmtime)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0_ckpt", type=str, default="checkpoints/run/p0_v2_step_202913", help="Phase 0 MLX Teacher checkpoint containing weak_decoder.safetensors")
    parser.add_argument("--ckpt_dir", type=str, default="", help="Path to student checkpoint dir (if empty, finds the latest in checkpoints/distilled)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", type=str, default="今天天气真不错，我们一起去", help="The text to encode into Z")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = args.device
    tokenizer = CharTokenizer()
    config = WeakDecoderConfig()

    ckpt_dir = args.ckpt_dir
    if not ckpt_dir:
        ckpt_dir = get_latest_student_ckpt("checkpoints/distilled")
        if not ckpt_dir:
            print("❌ 找不到任何学生模型的 Checkpoint！")
            return
    print(f"🔥 加载学生模型 (TinyCharEncoder): {ckpt_dir}")

    # 1. Load Student
    args_path = os.path.join(os.path.dirname(ckpt_dir), "training_args.json")
    if not os.path.exists(args_path):
        args_path = glob.glob(os.path.join(os.path.dirname(ckpt_dir), "*_args.json"))
        args_path = args_path[0] if args_path else None
        
    student_d_model = 1024
    student_n_heads = 8
    student_n_layers = 6
    if args_path and os.path.exists(args_path):
        with open(args_path, 'r') as f:
            targs = json.load(f)
            student_d_model = targs.get("student_d_model", 1024)
            student_n_heads = targs.get("student_n_heads", 8)
            student_n_layers = targs.get("student_n_layers", 6)

    student = TinyCharEncoderCUDA(
        vocab_size=config.vocab_size,
        d_model=student_d_model,
        n_heads=student_n_heads,
        n_layers=student_n_layers,
        z_dim=1024
    ).to(device)
    
    from safetensors.torch import load_file
    student_safe = os.path.join(ckpt_dir, "student.safetensors")
    student_pt = os.path.join(ckpt_dir, "student.pt")
    if os.path.exists(student_safe):
        student.load_state_dict(load_file(student_safe), strict=True)
    elif os.path.exists(student_pt):
        student.load_state_dict(torch.load(student_pt, map_location="cpu"), strict=True)
    else:
        print(f"❌ 找不到 {student_safe} 或 {student_pt}")
        return
    student.eval()

    # 2. Load Teacher Decoder
    print(f"🔥 加载老师解码器 (WeakDecoder): {args.p0_ckpt}")
    decoder = WeakDecoderCUDA(
        z_dim=1024,
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers
    ).to(device)
    
    decoder_safe = os.path.join(args.p0_ckpt, "weak_decoder.safetensors")
    load_mlx_safetensors_into_torch(decoder, decoder_safe)
    decoder.eval()

    print("\n" + "="*50)
    print(f"📝 输入文本 (Prompt): {args.prompt}")
    
    # 3. Encode to Z
    tokens, _ = tokenizer.encode_batch([args.prompt], max_len=256)
    tokens = tokens.to(device)
    
    with torch.no_grad():
        z_pred = student(tokens) # (1, 1024)
        
    print(f"✨ 成功提取 Z 向量! L2 Norm: {torch.norm(z_pred).item():.4f}, Std: {z_pred.std().item():.4f}")
    
    # 4. Generate
    start_token = tokenizer.char_to_id.get(args.prompt[-1], 0) if args.prompt else 0
    print(f"🚀 开始让 WeakDecoder 根据 Z 向量进行续写 (温度={args.temperature})...")
    
    with torch.no_grad():
        generated_ids = decoder.generate(
            z_pred, 
            start_token=start_token, 
            max_tokens=args.max_tokens, 
            temperature=args.temperature
        )
        
    result_text = tokenizer.decode(generated_ids)
    print("\n" + "="*50)
    print(f"🧠 Z-空间解码结果:\n{args.prompt[:-1]}{result_text}")
    print("="*50)

if __name__ == "__main__":
    main()
