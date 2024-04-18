import pandas as pd
from pmdarima.arima import ndiffs

from codebase.load_data.config import DATASETS

from codebase.evaluation.workflow import EvaluationWorkflow

ROPE = 5
DATA = ['M3', 'Tourism']

# n diffs

ndiffs_by_ids = {}
for dname in DATA:
    data_cls_ = DATASETS[dname]

    for group in data_cls_.data_group:
        ds = data_cls_.load_data(group)
        ds_grouped = ds.groupby('unique_id')

        for tsname, df_ in ds_grouped:
            print(tsname)
            s = df_['y'].values
            ndiffs_by_ids[f'{dname}_{tsname}'] = ndiffs(s, test='adf')

ndiffs_by_ids = pd.Series(ndiffs_by_ids).reset_index()
ndiffs_by_ids.columns = ['index', 'No. diffs']

#

results_df = EvaluationWorkflow.read_all_results(DATA)

eval_wf = EvaluationWorkflow(results_df, baseline='SNaive', reference='NHITS')

df = eval_wf.eval_by_series()

df = df.reset_index()

df = df.merge(ndiffs_by_ids, on='index', how='left').set_index('index')
df.groupby('No. diffs').mean().T
