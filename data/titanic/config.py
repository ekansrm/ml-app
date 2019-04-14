import os
import kaggle

competition = 'titanic'
file_kaggle_base = kaggle.api.get_config_value('data')
file_base = os.path.join(file_kaggle_base, competition)


file_train_csv = os.path.join(file_base, "train.csv")
file_test_csv = os.path.join(file_base, "test.csv")
file_gender_submission_csv = os.path.join(file_base, "gender_submission.csv")
