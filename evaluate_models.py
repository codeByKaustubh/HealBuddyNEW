import warnings

from src.config import DATA_PATH, RANDOM_STATE
from src.data import load_data
from src.models import train_models

warnings.filterwarnings("ignore")


def main():
    df = load_data(DATA_PATH)
    (
        _X,
        _feature_cols,
        target_col,
        _le,
        _fitted_models,
        _lime_explainer,
        eval_df,
        eval_notes,
    ) = train_models(df, RANDOM_STATE)

    print(f"Dataset: {DATA_PATH}")
    print(f"Target column: {target_col}")
    print()
    print("Model Evaluation")
    for _, row in eval_df.iterrows():
        print(f"- {row['Model']}")
        print(f"  Training Accuracy : {row['Training Accuracy']:.4f}")
        print(f"  Holdout Accuracy  : {row['Holdout Accuracy']:.4f}")
        print(f"  Holdout Macro F1  : {row['Holdout Macro F1']:.4f}")
        print(f"  CV Accuracy (mean): {row['CV Accuracy (mean)']:.4f}")
        print(f"  CV Macro F1 (mean): {row['CV Macro F1 (mean)']:.4f}")
        print()

    if eval_notes:
        print("Notes")
        for note in eval_notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
