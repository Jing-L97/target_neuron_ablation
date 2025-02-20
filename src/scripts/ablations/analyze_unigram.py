import sys
sys.path.append('../')
import pandas as pd
import plotly.express as px
from neuron_analyzer.ablations import *
import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import qualitative


def load_result(
    model_name:str,
    dataset:str = 'stas/c4-en-10k',
    data_range_start:int = 0,
    data_range_end:int = 500,
    k = 10
    ):
    save_path = f'./{output_dir}/{model_name}/unigram/{dataset.replace("/","_")}_{data_range_start}-{data_range_end}/k{k}.feather'
    final_df = pd.read_feather(save_path)
    # %%
    final_df['delta_loss'] = final_df['loss_post_ablation'] - final_df['loss']
    final_df['delta_loss_with_frozen_unigram'] = final_df['loss_post_ablation_with_frozen_unigram'] - final_df['loss']
    final_df['abs_delta_loss_post_ablation'] = np.abs(final_df['loss_post_ablation'] - final_df['loss'])
    final_df['abs_delta_loss_post_ablation_with_frozen_unigram'] = np.abs(final_df['loss_post_ablation_with_frozen_unigram'] - final_df['loss'])
    final_df['delta_entropy'] = final_df['entropy_post_ablation'] - final_df['entropy']
    if 'kl_divergence_before' in final_df.columns:
        print('kl_divergence_before found')
        final_df['kl_from_unigram_diff'] = final_df['kl_divergence_after'] - final_df['kl_divergence_before']
        final_df['kl_from_unigram_diff_with_frozen_unigram'] = final_df['kl_divergence_after_frozen_unigram'] - final_df['kl_divergence_before']
        final_df['abs_kl_from_unigram_diff'] = final_df['kl_from_unigram_diff'].abs()
    final_df['abs_kl_from_unigram_diff'] = final_df['kl_from_unigram_diff'].abs()

    return final_df