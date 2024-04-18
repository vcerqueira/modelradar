class LoadDataset:
    DATASET_PATH = '/Users/vcerq/Documents/datasets'
    DATASET_NAME = ''

    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
    }

    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
    }

    context_length = {
        'Yearly': 8,
        'Quarterly': 10,
        'Monthly': 24,
    }

    frequency_pd = {
        'Yearly': 'Y',
        'Quarterly': 'Q',
        'Monthly': 'M'
    }

    data_group = [*horizons_map]
    frequency = [*frequency_map.values()]
    horizons = [*horizons_map.values()]

    @classmethod
    def load_data(cls, group):
        pass
