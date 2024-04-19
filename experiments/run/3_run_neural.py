from neuralforecast.models import NHITS
from neuralforecast import NeuralForecast

from codebase.load_data.config import DATASETS

CONFIG = {
    'max_steps': 1500,
    'val_check_steps': 50,
    'enable_checkpointing': True,
    'start_padding_enabled': True,
    'accelerator': 'cpu'}

for data_name in DATASETS:
    for group in DATASETS[data_name].data_group:
        # if data_name != 'M4':# or group != "Yearly":
        #     continue

        print(data_name, group)

        data_cls = DATASETS[data_name]
        OUTPUT_DIR = f'assets/results/by_group/{data_name}_{group}_neural.csv'

        ds = data_cls.load_data(group)
        h = data_cls.horizons_map[group]
        n_lags = data_cls.context_length[group]
        freq = data_cls.frequency_pd[group]
        season_len = data_cls.frequency_map[group]

        models = [
            NHITS(h=h,
                  input_size=n_lags,
                  **CONFIG),
        ]

        nf = NeuralForecast(models=models, freq=freq)

        cv_nf = nf.cross_validation(df=ds, test_size=h, n_windows=None)

        cv_nf.to_csv(OUTPUT_DIR, index=False)
