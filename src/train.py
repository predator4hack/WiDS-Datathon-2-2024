import os
import argparse

import joblib
import pandas as pd
from sklearn import metrics

import config
import model_dispatcher

def run(fold, model):
    df = pd.read_csv(config.TRAINING_KFOLD_FILE)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(config.TRAGET)
    y_train = df_train[config.TRAGET].values

    x_valid = df_valid.drop(config.TRAGET)
    y_valid = df_valid[config.TRAGET].values

    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)
    accuracy = metrics.roc_auc_score(y_valid, preds)
    print(f"K Fold={fold}, Accuracy={accuracy}")


    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    args = parser.parse_args()
    
    run(
        fold=args.fold,
        model=args.model
    )