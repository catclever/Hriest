"""
【脚本功能】：Phase 0.5 (TinyCharEncoderCUDA) 学生的专属 MTEB 考场
【使用场景】：测试蒸馏出来的学生模型（纯字符输入）生成的 Z 向量在 LCQMC/BQ/AFQMC 等下游任务上的真实战力。
【用法】：python dev_tests/evaluate_mteb_student.py --ckpt_dir checkpoints/distilled/YOUR_STEP
"""
import os
import json
import glob
import argparse
import torch
import numpy as np
from tqdm import tqdm
import warnings

try:
    import mteb
except ImportError:
    pass

warnings.filterwarnings('ignore')

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from distilled_emb.model_cuda import TinyCharEncoderCUDA
from training.core.char_tokenizer import CharTokenizer
from model.config import WeakDecoderConfig

TASKS = ["LCQMC", "BQ", "AFQMC"]

class StudentEmbeddingWrapper:
    def __init__(self, ckpt_dir, device="cuda"):
        self.device = device
        self.tokenizer = CharTokenizer()
        self.max_seq_len = 256 # Default or read from args
        
        args_path = os.path.join(os.path.dirname(ckpt_dir), "training_args.json")
        if not os.path.exists(args_path):
            args_path = glob.glob(os.path.join(os.path.dirname(ckpt_dir), "*_args.json"))
            args_path = args_path[0] if args_path else None
            
        student_d_model = 1024
        student_n_heads = 8
        student_n_layers = 6
        
        if args_path and os.path.exists(args_path):
            with open(args_path, 'r') as f:
                args = json.load(f)
                student_d_model = args.get("student_d_model", 1024)
                student_n_heads = args.get("student_n_heads", 8)
                student_n_layers = args.get("student_n_layers", 6)
                self.max_seq_len = args.get("max_seq_len", 256)

        config = WeakDecoderConfig()
        self.model = TinyCharEncoderCUDA(
            vocab_size=config.vocab_size,
            d_model=student_d_model,
            n_heads=student_n_heads,
            n_layers=student_n_layers,
            z_dim=1024,
            max_seq_len=self.max_seq_len
        ).to(self.device)
        
        safetensor_path = os.path.join(ckpt_dir, "student.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(safetensor_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        
    def encode(self, sentences, batch_size=32, **kwargs):
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
                batch_texts = sentences[i:i+batch_size]
                tokens, _ = self.tokenizer.encode_batch(batch_texts, max_len=self.max_seq_len)
                tokens = tokens.to(self.device)
                
                z_pred = self.model(tokens)
                # Ensure the embedding is returned as float32 numpy array
                all_embeddings.append(z_pred.cpu().numpy().astype(np.float32))
                
        return np.concatenate(all_embeddings, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to student checkpoint dir (e.g. checkpoints/distilled/v2_step_10000)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--output_dir", type=str, default="./mteb_results_student")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🔥 加载学生模型从: {args.ckpt_dir}")
    model = StudentEmbeddingWrapper(args.ckpt_dir, device=args.device)
    
    active_tasks = mteb.get_tasks(tasks=TASKS)
    eval_engine = mteb.MTEB(tasks=active_tasks)
    
    safe_name = os.path.basename(os.path.normpath(args.ckpt_dir))
    output_folder = os.path.join(args.output_dir, f"student_{safe_name}")
    
    eval_engine.run(model, output_folder=output_folder, encode_kwargs={"batch_size": 256})
    
    print(f"\n🏆 [绝密考核榜单] - {safe_name}")
    print("=" * 60)
    for task in TASKS:
        files = glob.glob(os.path.join(output_folder, f"{task}*.json"))
        if not files:
            files = glob.glob(os.path.join(output_folder, "**", f"{task}*.json"), recursive=True)
        if not files: 
            print(f"   ├─ 任务 {task}: [缺考/漏考]")
            continue
        
        latest_file = max(files, key=os.path.getmtime)
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        scores = data.get("scores", {})
        test_scores = scores.get("test", [{}])[0] if "test" in scores else scores.get("validation", [{}])[0]
        if not test_scores: 
            test_scores = data.get("test", {})
        
        metric = 0.0
        if "accuracy" in test_scores: 
            metric = test_scores["accuracy"]
        elif "cos_sim" in test_scores and "spearman" in test_scores["cos_sim"]: 
            metric = test_scores["cos_sim"]["spearman"]
        elif "spearman" in test_scores: 
            metric = test_scores["spearman"]
            
        print(f"   ├─ 任务 {task}: 战力得分 {metric*100:.2f} 分")
    print("-" * 60)

if __name__ == "__main__":
    main()
