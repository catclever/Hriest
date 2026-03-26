import argparse
import numpy as np
import mlx.core as mx
import os

try:
    import mteb
except ImportError:
    print("打榜前请务必先安装依赖： pip install mteb[zh] numpy")
    import sys
    sys.exit(1)

from training.char_tokenizer import CharTokenizer
from model.config import ModelConfig
from distilled_emb.model import TinyCharEncoder

class TinyMTEBWrapper:
    """
    将我们自研的 MLX TinyCharEncoder 包装成标准的 MTEB 调用接口。
    """
    def __init__(self, model, tokenizer, max_seq_len=512):
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
                
            x_mx = mx.array(padded_ids)
            mask_mx = mx.array(masks)
            
            # Predict & Convert back to flat numpy
            z_pred_mx = self.model(x_mx, mask_mx)
            mx.eval(z_pred_mx) # Force array evaluation
            all_embs.append(np.array(z_pred_mx))
            
        return np.concatenate(all_embs, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Standalone C-MTEB Evaluator")
    parser.add_argument("--ckpt", type=str, required=True, help="训练好的 TinyBERT checkpoint 路径 (例如 checkpoints/distilled/tinybert_v1_step_50000)")
    parser.add_argument("--tasks", type=str, nargs="+", default=["LCQMC", "BQCorpus", "AFQMC"], help="想要测的 C-MTEB 任务名")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    # 1. 初始化相同的架构
    config = ModelConfig()
    tokenizer = CharTokenizer()
    
    print(f"Loading NativeCharEncoder Weights from {args.ckpt}...")
    model = TinyCharEncoder(
        vocab_size=config.vocab_size,
        d_model=1024,
        n_heads=8,
        n_layers=6,
        max_seq_len=config.max_seq_len,
        z_dim=config.z_dim
    )
    
    # Load MLX explicit safetensor mapped weights
    model_path = f"{args.ckpt}/student.safetensors"
    if not os.path.exists(model_path):
        print(f"[错误] 在 {args.ckpt} 下没找到 student.safetensors。请确保训练已经保存！")
        return
        
    model.load_weights(model_path)
    mx.eval(model.parameters())
    print("模型加载完成。极其轻量级，即将开启火力！")

    # 2. 包装模型并启动 MTEB 霸榜
    fast_wrapper = TinyMTEBWrapper(model, tokenizer, max_seq_len=config.max_seq_len)
    
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
