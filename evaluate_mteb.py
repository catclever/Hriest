import argparse
import numpy as np
import os
import json
import sys

try:
    import mteb
except ImportError:
    print("打榜前请务必先安装依赖： pip install mteb[zh] numpy")
    import sys
    sys.exit(1)

# Avoid local /home/Hriest/model.py shadowing /home/HollyShit/model package.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]

from training.core.char_tokenizer import CharTokenizer
from model.config import ModelConfig
from model_cuda import TinyCharEncoderCUDA

class TinyMTEBWrapperMLX:
    """
    将我们自研的 MLX TinyCharEncoder 包装成标准的 MTEB 调用接口。
    """
    def __init__(self, model, tokenizer, max_seq_len=512):
        import mlx.core as mx
        self.mx = mx
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    @property
    def mteb_model_meta(self):
        class MockMeta:
            name = "TinyCharEncoder"
            revision = "1.0.0"
            release_date = "2026-03-25"
            languages = ["cmn"]
            framework = []
            
            def model_name_as_path(self):
                return "TinyCharEncoder"
                
            def to_dict(self):
                return {
                    "name": self.name, 
                    "revision": self.revision,
                    "languages": self.languages,
                    "release_date": self.release_date
                }
            def model_copy(self, update=None):
                if update:
                    for k, v in update.items():
                        setattr(self, k, v)
                return self
                
            def __getattr__(self, name):
                return None
                
        return MockMeta()
        
    def similarity(self, embeddings1, embeddings2):
        emb1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        emb2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        return emb1 @ emb2.T

    def similarity_pairwise(self, embeddings1, embeddings2):
        emb1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        emb2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        return np.sum(emb1 * emb2, axis=1)
        
    def encode(self, sentences, batch_size=256, **kwargs):
        """
        MTEB 官方接口：输入字符串列表，输出 np.ndarray 的向量数组。
        """
        # 兼容 MTEB 2.11 可能会传入 DataLoader 的诡异操作
        if type(sentences).__name__ == "DataLoader":
            flat_sentences = []
            for batch in sentences:
                if isinstance(batch, (list, tuple)): flat_sentences.extend(batch)
                elif isinstance(batch, dict): flat_sentences.extend(list(batch.values())[0])
                else: flat_sentences.append(batch)
            sentences = flat_sentences
        elif not isinstance(sentences, list):
            sentences = list(sentences)
            
        all_embs = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = [self.tokenizer.encode(t, add_special_tokens=True)[:self.max_seq_len] for t in batch]
            
            # Dynamic Batch Padding
            max_len = max(len(seq) for seq in encoded)
            padded_ids = []
            masks = []
            
            for seq in encoded:
                pad_len = max_len - len(seq)
                padded_ids.append(seq + [self.tokenizer.pad_token_id] * pad_len)
                masks.append([1] * len(seq) + [0] * pad_len)
                
            x_mx = self.mx.array(padded_ids)
            mask_mx = self.mx.array(masks)
            
            # Predict & Convert back to flat numpy
            z_pred_mx = self.model(x_mx, mask_mx)
            self.mx.eval(z_pred_mx) # Force array evaluation
            all_embs.append(np.array(z_pred_mx))
            
        return np.concatenate(all_embs, axis=0)


class TinyMTEBWrapperTorch:
    def __init__(self, model, tokenizer, max_seq_len=512):
        import torch
        self.torch = torch
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = next(model.parameters()).device
    
    @property
    def mteb_model_meta(self):
        class MockMeta:
            name = "TinyCharEncoderCUDA"
            revision = "1.0.0"
            release_date = "2026-03-25"
            languages = ["cmn"]
            framework = []
            def model_name_as_path(self):
                return "TinyCharEncoderCUDA"
            def to_dict(self):
                return {
                    "name": self.name,
                    "revision": self.revision,
                    "languages": self.languages,
                    "release_date": self.release_date
                }
            def model_copy(self, update=None):
                if update:
                    for k, v in update.items():
                        setattr(self, k, v)
                return self
            def __getattr__(self, name):
                return None
        return MockMeta()
    
    def similarity(self, embeddings1, embeddings2):
        emb1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        emb2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        return emb1 @ emb2.T
    
    def similarity_pairwise(self, embeddings1, embeddings2):
        emb1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        emb2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        return np.sum(emb1 * emb2, axis=1)
    
    def encode(self, sentences, batch_size=256, **kwargs):
        if type(sentences).__name__ == "DataLoader":
            flat_sentences = []
            for batch in sentences:
                if isinstance(batch, (list, tuple)):
                    flat_sentences.extend(batch)
                elif isinstance(batch, dict):
                    flat_sentences.extend(list(batch.values())[0])
                else:
                    flat_sentences.append(batch)
            sentences = flat_sentences
        elif not isinstance(sentences, list):
            sentences = list(sentences)
        
        all_embs = []
        with self.torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                encoded = [self.tokenizer.encode(t, add_special_tokens=True)[:self.max_seq_len] for t in batch]
                max_len = max(len(seq) for seq in encoded)
                padded_ids = []
                masks = []
                for seq in encoded:
                    pad_len = max_len - len(seq)
                    padded_ids.append(seq + [self.tokenizer.pad_token_id] * pad_len)
                    masks.append([1] * len(seq) + [0] * pad_len)
                token_inputs = self.torch.tensor(padded_ids, dtype=self.torch.long, device=self.device)
                attention_mask = self.torch.tensor(masks, dtype=self.torch.float32, device=self.device)
                z_pred = self.model(token_inputs, attention_mask)
                all_embs.append(z_pred.cpu().numpy())
        return np.concatenate(all_embs, axis=0)


def load_student_dims_from_args(ckpt_dir: str):
    args_path = os.path.join(ckpt_dir, "training_args.json")
    d_model, n_heads, n_layers = 1024, 8, 6
    if os.path.exists(args_path):
        with open(args_path, "r", encoding="utf-8") as f:
            train_args = json.load(f)
        d_model = int(train_args.get("student_d_model", d_model))
        n_heads = int(train_args.get("student_n_heads", n_heads))
        n_layers = int(train_args.get("student_n_layers", n_layers))
    return d_model, n_heads, n_layers

def main():
    parser = argparse.ArgumentParser(description="Standalone C-MTEB Evaluator")
    parser.add_argument("--ckpt", type=str, required=True, help="训练好的 TinyBERT checkpoint 路径 (例如 checkpoints/distilled/tinybert_v1_step_50000)")
    parser.add_argument("--tasks", type=str, nargs="+", default=["LCQMC", "BQCorpus", "AFQMC"], help="想要测的 C-MTEB 任务名")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="仅用于 .pt 模型评测")
    args = parser.parse_args()

    # 1. 初始化与训练一致的架构
    config = ModelConfig()
    tokenizer = CharTokenizer()
    d_model, n_heads, n_layers = load_student_dims_from_args(args.ckpt)
    
    pt_path = os.path.join(args.ckpt, "student.pt")
    mlx_path = os.path.join(args.ckpt, "student.safetensors")
    if os.path.exists(pt_path):
        import torch
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        else:
            device = torch.device(args.device)
        print(f"Loading PyTorch TinyCharEncoder from {pt_path} on {device} ...")
        model = TinyCharEncoderCUDA(
            vocab_size=config.vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=config.max_seq_len,
            z_dim=config.z_dim
        ).to(device)
        state = torch.load(pt_path, map_location=device)
        model.load_state_dict(state, strict=True)
        fast_wrapper = TinyMTEBWrapperTorch(model, tokenizer, max_seq_len=config.max_seq_len)
    elif os.path.exists(mlx_path):
        import mlx.core as mx
        from distilled_emb.model import TinyCharEncoder
        print(f"Loading MLX TinyCharEncoder from {mlx_path} ...")
        model = TinyCharEncoder(
            vocab_size=config.vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=config.max_seq_len,
            z_dim=config.z_dim
        )
        model.load_weights(mlx_path)
        mx.eval(model.parameters())
        fast_wrapper = TinyMTEBWrapperMLX(model, tokenizer, max_seq_len=config.max_seq_len)
    else:
        print(f"[错误] 在 {args.ckpt} 下没找到 student.pt 或 student.safetensors。")
        return
    
    print("模型加载完成。即将启动评测。")
    
    print(f"\n=============================================")
    print(f"开始启动 C-MTEB 评测任务：{args.tasks}")
    print(f"测试类型侧重：深度语义匹配 (STS), 同反义复述判断")
    print(f"=============================================\n")

    # Note: C-MTEB tasks run automatically using HuggingFace datasets downloaded under the hood
    
    # 适配最新 MTEB v2.11+ API
    active_tasks = mteb.get_tasks(tasks=args.tasks)
    eval_engine = mteb.MTEB(tasks=active_tasks)
    
    # mteb .run() 传入模型 wrapper
    results = eval_engine.run(fast_wrapper, output_folder="distilled_emb/mteb_results", encode_kwargs={"batch_size": args.batch_size})
    
    print(f"\n=============================================")
    print(f"打榜结束！成绩已经详细保存到了 distilled_emb/mteb_results 目录下。")
    print(f"正在为您从 JSON 中提取战报纲要...")
    print(f"=============================================\n")
    
    import json
    import glob
    
    print(f"================== 📊 最终大考天梯榜 📊 ==================")
    for task_name in args.tasks:
        # mteb usually outputs to <folder>/<model_name>/<task>.json. Let's hunt it recursively.
        search_pattern = f"distilled_emb/mteb_results/**/{task_name}*.json"
        
        files = glob.glob(search_pattern, recursive=True)
        if not files:
            print(f" [!] 未找到 {task_name} 的结果文件。")
            continue
            
        # Get the latest result
        latest_file = max(files, key=os.path.getmtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # It's usually under "test" split, but MTEB 2.11+ uses 'validation' for AFQMC
        scores_dict = data.get("scores", {})
        test_scores = scores_dict.get("test", [{}])[0] if "test" in scores_dict else {}
        if not test_scores:
            test_scores = scores_dict.get("validation", [{}])[0] if "validation" in scores_dict else {}
        if not test_scores:
            test_scores = data.get("test", {})
            
        main_score = 0.0
        score_name = "N/A"
        
        # Smart extraction logic covering both STS and PairClassification
        if "accuracy" in test_scores:
            main_score = test_scores["accuracy"]
            score_name = "Accuracy (准确率)"
        elif "cos_sim" in test_scores and "spearman" in test_scores["cos_sim"]:
            main_score = test_scores["cos_sim"]["spearman"]
            score_name = "Cosine Spearman (语义斯皮尔曼相关系数)"
        elif "spearman" in test_scores:
            main_score = test_scores["spearman"]
            score_name = "Spearman (斯皮尔曼)"
        elif "f1" in test_scores:
            main_score = test_scores["f1"]
            score_name = "F1 Score"
        else:
            # Deep hunt
            for k, v in test_scores.items():
                if isinstance(v, float):
                    main_score = v
                    score_name = k
                    break
                elif isinstance(v, dict):
                    if "accuracy" in v:
                        main_score = v["accuracy"]
                        score_name = f"{k}.accuracy"
                        break
                    elif "spearman" in v:
                        main_score = v["spearman"]
                        score_name = f"{k}.spearman"
                        break
                    elif "f1" in v:
                        main_score = v["f1"]
                        score_name = f"{k}.f1"
                        break
                        
        display_score = main_score * 100 if main_score <= 1.0 else main_score
        
        # Color coding grading (Zero-shot scale)
        if display_score >= 85.0:
            grade = "🌟 神迹 (零样本硬刚业界 SOTA 大模型)"
        elif display_score >= 80.0:
            grade = "💎 可以实用 (物理降维大成功，商业可用)"
        elif display_score >= 70.0:
            grade = "🔥 泛化不错 (挺过脏数据跨域压力测试)"
        elif display_score >= 60.0:
            grade = "✅ 预料之内 (纯正规维基百科的尽力局)"
        else:
            grade = "💀 特别不行 (模型变废柴了，得查查哪漏水了)"
            
        print(f" 🎯 任务卷子 [{task_name.ljust(10)}] | {score_name.ljust(35)} = {display_score:05.2f}% | {grade}")
        
    print(f"==========================================================\n")

if __name__ == "__main__":
    main()
