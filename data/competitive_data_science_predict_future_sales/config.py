import os
import kaggle

competition = 'competitive-data-science-predict-future-sales'
file_kaggle_base = kaggle.api.get_config_value('data')
file_base = os.path.join(file_kaggle_base, competition)

file_sales_train_csv_gz = os.path.join(file_base, "sales_train.csv.gz")
file_sample_submission_csv_gz = os.path.join(file_base, "sample_submission.csv.gz")
file_test_csv_gz = os.path.join(file_base, "test.csv.gz")

file_item_categories_csv = os.path.join(file_base, "item_categories.csv")
file_items_csv = os.path.join(file_base, "items.csv")
file_sales_train_v2_csv = os.path.join(file_base, "sales_train_v2.csv")
file_sample_submission_csv = os.path.join(file_base, "sample_submission.csv")
file_shops_csv = os.path.join(file_base, "shops.csv")
file_test_csv = os.path.join(file_base, "test.csv")


# 下载文件

# 解压文件

