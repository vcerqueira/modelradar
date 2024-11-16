from codebase.load_data.m4 import M4Dataset
from codebase.load_data.m3 import M3Dataset
from codebase.load_data.tourism import TourismDataset

DATASETS = {
    'Tourism': TourismDataset,
    'M3': M3Dataset,
    'M4': M4Dataset,
}
