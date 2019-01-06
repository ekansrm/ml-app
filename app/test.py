import os
import kaggle

competition = 'competitive_data_science_predict_future_sales'
config_key_data_path_path = 'data'


if __name__ == '__main__':

    data_path_base = kaggle.api.get_config_value(config_key_data_path_path)
    data_path = os.path.join(data_path_base, competition)


