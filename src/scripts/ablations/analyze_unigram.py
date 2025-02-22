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



def select_top_token_frequency_neurons(
    final_df: pd.DataFrame, 
    unigram_kl_threshold: float = 2.0, 
    unigram_mediation_threshold: float = 0.5, 
    top_n: int = 10
) -> dict[str, list[str]]:
    """
    Correctly select top token frequency neurons based on multiple criteria.
    """
    # Calculate the mediation effect
    final_df['mediation_effect'] = (
        1 - final_df['abs_delta_loss_post_ablation_with_frozen_unigram'] 
        / final_df['abs_delta_loss_post_ablation']
    )

    ranked_neurons = final_df.sort_values(
        by='mediation_effect', 
        ascending=False
    )
    
    # Select top N neurons, preserving the original sorting
    top_neurons = ranked_neurons['component_name'].head(top_n).tolist()

    return {
        model_name: top_neurons
    }

def aggregate_result(final_df:pd.DataFrame,unigram_neurons_dict:dict)->pd.DataFrame:

    unigram_neurons = unigram_neurons_dict.get(model_name, [])
    final_df['is_unigram'] = final_df['component_name'].isin(unigram_neurons).astype(bool)
    
    columns_to_aggregate =list(final_df.columns[8:]) + ['loss']
    print(columns_to_aggregate)
    agg_results = final_df[columns_to_aggregate].groupby('component_name').mean().reset_index()

    # make scatter plot of delta_loss and delta_loss_with_frozen_unigram for each neuron
    agg_results['delta_loss-delta_loss_with_frozen_unigram'] = agg_results['delta_loss'] - agg_results['delta_loss_with_frozen_unigram']
    agg_results['abs_delta_loss-abs_delta_loss_with_frozen_unigram'] = agg_results['abs_delta_loss_post_ablation'] - agg_results['abs_delta_loss_post_ablation_with_frozen_unigram']
    # %%
    # make scatter plot of delta_loss and delta_loss_with_frozen_unigram for each neuron
    agg_results['delta_loss-delta_loss_with_frozen_unigram'] = agg_results['delta_loss'] - agg_results['delta_loss_with_frozen_unigram']
    agg_results['abs_delta_loss-abs_delta_loss_with_frozen_unigram'] = agg_results['abs_delta_loss_post_ablation'] - agg_results['abs_delta_loss_post_ablation_with_frozen_unigram']
    agg_results['1-abs_delta_loss_with_frozen_unigram/abs_delta_loss'] = 1 - agg_results['abs_delta_loss_post_ablation_with_frozen_unigram'] / agg_results['abs_delta_loss_post_ablation']

    return agg_results
    
def plot_top_token_frequency_neurons(
    agg_results: pd.DataFrame, 
    unigram_neurons: list[str], 
    model_name: str
):  # Consider specifying the exact return type of your plotting library
    """Create a scatter plot highlighting top token frequency neurons."""
    

    # Prepare neuron type column
    conditions = [(agg_results['is_unigram'] == True)]
    choices = ['Token Frequency']
    agg_results['Neuron Type'] = np.select(conditions, choices, default='Normal')

    return fig

