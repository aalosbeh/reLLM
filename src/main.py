import argparse
from pathlib import Path

from utils import load_data, preprocess, train_test_split_df
from train import train_model
from evaluate import evaluate


def main(args):
    df = load_data(args.data_path)
    df, le = preprocess(df)
    train_df, test_df = train_test_split_df(df)

    trainer, tokenizer = train_model(
        args.model,
        train_df['text'].tolist(),
        train_df['label_id'].tolist(),
        test_df['text'].tolist(),
        test_df['label_id'].tolist(),
        output_dir=args.output_dir,
    )

    # Evaluate
    acc, f1, report = evaluate(trainer, test_df['label_id'].tolist())

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(report)

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    with open(Path(args.output_dir) / "metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GPT model on CTC data")
    parser.add_argument("--data-path", required=True, help="Path to ctc_data.csv")
    parser.add_argument("--model", default="EleutherAI/gpt-neo-125M", help="Model name")
    parser.add_argument("--output-dir", default="./model", help="Directory to save model and metrics")
    args = parser.parse_args()
    main(args)
