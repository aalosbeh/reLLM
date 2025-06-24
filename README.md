# Drone CTC GPT Fine-tuning

This repository contains code for comparing GPT model performance before and after fine-tuning on drone-collected cover time channel (CTC) data. The dataset is categorized into two groups:

- `group_1`: collected when the drone was near a laptop
- `group_2`: collected when the drone was far from the laptop

The project demonstrates preprocessing, training an open-source GPT model, fine-tuning for classification, and evaluating results.

## Directory Structure

```
reLLM/
├── data/                # Place `ctc_data.csv` here
├── paper/               # LaTeX research paper
├── src/
│   ├── main.py          # Entry point
│   ├── train.py         # Training utilities
│   ├── evaluate.py      # Evaluation scripts
│   └── utils.py         # Helper functions
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup

1. Create a Python 3.10 environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download or place the dataset `ctc_data.csv` inside the `data/` directory. The CSV should contain a `text` column with CTC samples and a `label` column with values `group_1` or `group_2`.

## Usage

Run training and evaluation from the command line:

```bash
python src/main.py --data-path data/ctc_data.csv --model EleutherAI/gpt-neo-125M
```

Results (accuracy, F1-score) will be printed and saved for the research paper in `paper/`.

This code can be executed in PyCharm or Jupyter Notebook. For notebooks, import the modules from `src/` and follow the same steps.
