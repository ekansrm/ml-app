import gc
import pandas as pd
import numpy as np
from app.k.pubg.data import FILE_DATA_SUBMISSION, FILE_DATA_TEST, FILE_DATA_TRAIN
from app.k.pubg.BaseLineLightGBM import feature_engineering, reduce_mem_usage, run_lgb


def train():
    x_train, y_train, train_columns, _ = feature_engineering(True, False)
    x_test, _, _, test_idx = feature_engineering(False, True)
    x_train = reduce_mem_usage(x_train)
    x_test = reduce_mem_usage(x_test)

    train_index = round(int(x_train.shape[0] * 0.8))
    dev_X = x_train[:train_index]
    val_X = x_train[train_index:]
    dev_y = y_train[:train_index]
    val_y = y_train[train_index:]
    gc.collect()
    # Training the model #
    pred_test, model = run_lgb(dev_X, dev_y, val_X, val_y, x_test)

    df_sub = pd.read_csv(FILE_DATA_SUBMISSION)
    df_test = pd.read_csv(FILE_DATA_TEST)
    df_sub['winPlacePerc'] = pred_test
    # Restore some columns
    df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")

    # Sort, rank, and assign adjusted ratio
    df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()
    df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()
    df_sub_group = df_sub_group.merge(
        df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(),
        on="matchId", how="left")
    df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)

    df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
    df_sub["winPlacePerc"] = df_sub["adjusted_perc"]

    # Deal with edge cases
    df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0
    df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1

    # Align with maxPlace
    # Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
    subset = df_sub.loc[df_sub.maxPlace > 1]
    gap = 1.0 / (subset.maxPlace.values - 1)
    new_perc = np.around(subset.winPlacePerc.values / gap) * gap
    df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc

    # Edge case
    df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0
    assert df_sub["winPlacePerc"].isnull().sum() == 0

    df_sub[["Id", "winPlacePerc"]].to_csv("submission_adjusted.csv", index=False)


if __name__ == '__main__':
    train()
