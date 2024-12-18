from exp_utils.load_data.m3 import M3Dataset
from exp_utils.load_data.tourism import TourismDataset
from exp_utils.load_data.gluonts import GluontsDataset

DATASETS = {
    'M3': M3Dataset,
    'Tourism': TourismDataset,
    'Gluonts': GluontsDataset,
}

DATA_GROUPS = [
    ('Gluonts', 'm1_monthly'),
    ('Gluonts', 'm1_quarterly'),
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
]
