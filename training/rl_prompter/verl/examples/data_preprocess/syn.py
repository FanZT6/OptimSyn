import argparse
import os
import datasets
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset_path", required=True, help="Path to the raw jsonl dataset file")
    parser.add_argument("--local_dir", default="~/data/book_rubrics", help="Local directory to save processed data")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the result")
    parser.add_argument("--domain", required=True, help="Domain value for processing")
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                        help="Ratio of data to randomly sample from the raw dataset (0 < r <= 1.0)")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    args.local_dir = os.path.expanduser(args.local_dir)

    user_prompt_template_path = "/mnt/rl/zz/SynAgent/prompt/generate_rubrics.txt"
    system_prompt_template_path = "/mnt/rl/zz/SynAgent/system_prompt/generate_rubrics.txt"

    with open(user_prompt_template_path, "r", encoding="utf-8") as f:
        user_prompt_template = f.read()

    with open(system_prompt_template_path, "r", encoding="utf-8") as f:
        system_prompt_template = f.read()

    # 读取原始数据集
    dataset = datasets.load_dataset("json", data_files=args.raw_dataset_path, split="train")

    # 如果采样比例 < 1.0 则先随机采样
    if args.sample_ratio < 1.0:
        if args.sample_ratio <= 0 or args.sample_ratio > 1:
            raise ValueError("--sample_ratio must be between 0 and 1.0")
        orig_len = len(dataset)
        dataset = dataset.train_test_split(
            test_size=1 - args.sample_ratio,
            seed=args.sample_seed
        )["train"]
        print(f"Sampled {len(dataset)} examples out of {orig_len} ({args.sample_ratio*100:.1f}%)")

    def make_map_fn(domain, raw_dataset_path):
        def process_fn(example, idx):
            # title = example["title"]
            content = example["text"]
            # content = example["content"]

            user_prompt = user_prompt_template.replace("{document}", content).replace("{domain}", domain)
            system_prompt = system_prompt_template.replace("{domain}", domain)

            return {
                "data_source": raw_dataset_path,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "ability": domain,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ""
                },
                "extra_info": {
                    "seed_data": content,
                    "domain": domain
                }
            }
        return process_fn

    processed_dataset = dataset.map(
        function=make_map_fn(args.domain, args.raw_dataset_path),
        with_indices=True
    )

    # Split into train (90%) and test (10%) randomly
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    print(train_dataset[0])

    # Save parquet files
    os.makedirs(args.local_dir, exist_ok=True)
    train_path = os.path.join(args.local_dir, f"train_{args.sample_ratio}.parquet")
    test_path = os.path.join(args.local_dir, f"test_{args.sample_ratio}.parquet")
    print(f"After split: train={len(train_dataset)}, test={len(test_dataset)}")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Data processing completed.")
    print(f"Train set saved to: {train_path}")
    print(f"Test set saved to: {test_path}")
