# %%
import os
import sys
sys.path.append('../')
import numpy as np
import torch
from datasets import load_dataset
from utils import get_entropy, load_model_from_tl_name, filter_entropy_activation_df, get_entropy_activation_df, get_pile_unigram_distribution
import neel.utils as nutils
import transformer_lens.utils as utils
import tqdm
import pathlib
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.nn.functional import kl_div
import gc

def log_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def process_batch_in_micro_chunks(model, res_stream, activation_deltas, neuron_indices, 
                                layer_idx, inp, unigram_direction_vocab, 
                                unigram_projection_values, log_unigram_distrib, device, 
                                micro_chunk_size=2):
    """Process a batch in very small chunks to minimize memory usage"""
    
    results = {
        'loss': [], 'entropy': [], 
        'loss_frozen': [], 'entropy_frozen': [],
        'kl_div': [], 'kl_div_frozen': []
    }
    
    for i in range(0, len(neuron_indices), micro_chunk_size):
        try:
            indices_chunk = neuron_indices[i:i+micro_chunk_size]
            
            # Calculate residual deltas for micro chunk
            deltas_chunk = activation_deltas[:, indices_chunk].unsqueeze(-1)
            deltas_chunk = deltas_chunk * model.W_out[layer_idx, indices_chunk, :]
            deltas_chunk = deltas_chunk.permute(1, 0, 2)
            
            # Process micro chunk
            res_stream_chunk = res_stream.repeat(len(indices_chunk), 1, 1)
            updated_stream = res_stream_chunk + deltas_chunk
            del res_stream_chunk, deltas_chunk
            
            # Apply layer norm and get logits
            normalized = model.ln_final(updated_stream)
            del updated_stream
            logits = normalized @ model.W_U + model.b_U
            del normalized
            
            # Calculate metrics
            loss = model.loss_fn(logits, inp.repeat(len(indices_chunk), 1), per_token=True).cpu()
            results['loss'].append(np.concatenate((loss, np.zeros((loss.shape[0], 1))), axis=1))
            
            entropy = get_entropy(logits).cpu()
            results['entropy'].append(entropy)
            
            # Calculate KL divergence
            logprobs = logits.log_softmax(dim=-1)
            kl = kl_div(logprobs, log_unigram_distrib, reduction='none', log_target=True).sum(axis=-1).cpu().numpy()
            results['kl_div'].append(kl)
            
            # Calculate frozen unigram projection
            logits_frozen = adjust_vectors_3dim(logits.detach(), unigram_direction_vocab, unigram_projection_values)
            
            loss_frozen = model.loss_fn(logits_frozen, inp.repeat(len(indices_chunk), 1), per_token=True).cpu()
            results['loss_frozen'].append(np.concatenate((loss_frozen, np.zeros((loss_frozen.shape[0], 1))), axis=1))
            
            entropy_frozen = get_entropy(logits_frozen).cpu()
            results['entropy_frozen'].append(entropy_frozen)
            
            logprobs_frozen = logits_frozen.log_softmax(dim=-1)
            kl_frozen = kl_div(logprobs_frozen, log_unigram_distrib, reduction='none', log_target=True).sum(axis=-1).cpu().numpy()
            results['kl_div_frozen'].append(kl_frozen)
            
            # Clean up
            del logits, logits_frozen, logprobs, logprobs_frozen
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at micro chunk {i}, reducing chunk size")
                torch.cuda.empty_cache()
                if micro_chunk_size > 1:
                    micro_chunk_size = max(1, micro_chunk_size // 2)
                    i -= micro_chunk_size
                    continue
                else:
                    raise RuntimeError("Cannot reduce chunk size further")
            raise e
            
    # Concatenate results
    return {k: np.concatenate(v, axis=0) if v else None for k, v in results.items()}

def adjust_vectors_3dim(v, u, target_values):
    """Memory efficient vector adjustment"""
    projections = (v @ u.unsqueeze(-1)).squeeze(-1)
    delta = target_values - projections
    return v + delta.unsqueeze(-1) * u

def mean_ablate_components(components_to_ablate=None,
                         unigram_distrib=None,
                         tokenized_data=None,
                         entropy_df=None,
                         model=None,
                         k=10,
                         device='cuda',
                         chunk_size=5):  # Reduced default chunk size
    
    torch.cuda.empty_cache()
    gc.collect()
    log_memory()
    
    # Sample sequences
    random_indices = np.random.choice(entropy_df.batch.unique(), k, replace=False)
    filtered_df = entropy_df[entropy_df.batch.isin(random_indices)].copy()
    
    print(f'ablate_components: ablate {len(components_to_ablate)} neurons with k = {k}')
    pbar = tqdm.tqdm(total=k, file=sys.stdout)
    
    results = {}
    activation_means = torch.tensor(entropy_df[[f'{c}_activation' for c in components_to_ablate]].mean())
    
    # Prepare unigram direction
    unigram_dir = unigram_distrib.log() - unigram_distrib.log().mean()
    unigram_dir /= unigram_dir.norm()
    
    # Get indices
    neuron_indices = [int(n.split('.')[1]) for n in components_to_ablate]
    layer_idx = int(components_to_ablate[0].split('.')[0])
    
    for batch_n in filtered_df.batch.unique():
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            # Process sequence
            tok_seq = tokenized_data['tokens'][batch_n]
            inp = tok_seq.unsqueeze(0).to(device)
            
            # Get initial model outputs
            model.reset_hooks()
            logits, cache = model.run_with_cache(inp)
            res_stream = cache[utils.get_act_name("resid_post", layer_idx)][0]
            prev_activation = cache[utils.get_act_name("post", layer_idx)][0, :, neuron_indices]
            del cache
            
            # Calculate activation deltas
            activation_deltas = activation_means.to(prev_activation.device) - prev_activation
            
            # Process in micro chunks
            batch_results = process_batch_in_micro_chunks(
                model=model,
                res_stream=res_stream,
                activation_deltas=activation_deltas,
                neuron_indices=neuron_indices,
                layer_idx=layer_idx,
                inp=inp,
                unigram_direction_vocab=unigram_dir,
                unigram_projection_values=logits @ unigram_dir,
                log_unigram_distrib=unigram_distrib.log(),
                device=device
            )
            
            # Create dataframe
            final_df = None
            for i, component_name in enumerate(components_to_ablate):
                df_chunk = filtered_df[filtered_df.batch == batch_n].copy()
                
                # Clean columns
                drop_cols = [f'{n}_activation' for n in components_to_ablate if n != component_name]
                df_chunk = df_chunk.drop(columns=drop_cols)
                df_chunk = df_chunk.rename(columns={f'{component_name}_activation': 'activation'})
                
                # Add metrics
                df_chunk['component_name'] = component_name
                df_chunk['loss_post_ablation'] = batch_results['loss'][i]
                df_chunk['loss_post_ablation_with_frozen_unigram'] = batch_results['loss_frozen'][i]
                df_chunk['entropy_post_ablation'] = batch_results['entropy'][i]
                df_chunk['entropy_post_ablation_with_frozen_unigram'] = batch_results['entropy_frozen'][i]
                df_chunk['kl_divergence_after'] = batch_results['kl_div'][i]
                df_chunk['kl_divergence_after_frozen_unigram'] = batch_results['kl_div_frozen'][i]
                
                final_df = df_chunk if final_df is None else pd.concat([final_df, df_chunk])
                del df_chunk
            
            results[batch_n] = final_df
            del final_df, batch_results
            gc.collect()
            torch.cuda.empty_cache()
            
            pbar.update(1)
            
        except Exception as e:
            print(f"\nError processing batch {batch_n}: {str(e)}")
            continue
    
    return results

@hydra.main(config_path='./conf', config_name='config_unigram_ablations', version_base="1.1")
def run_and_store_ablation_results(args: DictConfig):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)
    
    # Memory optimization settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Setup directories
    os.chdir(args.chdir)
    save_path = f'./{args.output_dir}/{args.model}/unigram/{args.dataset.replace("/","_")}_{args.data_range_start}-{args.data_range_end}'
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        with open(args.hf_token_path, 'r') as f:
            hf_token = f.read()
        
        model, tokenizer = load_model_from_tl_name(
            args.model, args.device, args.transformers_cache_dir, hf_token=hf_token
        )
        model = model.to(args.device)
        model.eval()
        
        # Load data
        data = load_dataset(args.dataset, split='train')
        first_1k = data.select(range(args.data_range_start, args.data_range_end))
        tokenized_data = utils.tokenize_and_concatenate(
            first_1k, tokenizer, max_length=256, column_name='text'
        )
        tokenized_data = tokenized_data.shuffle(args.seed)
        token_df = nutils.make_token_df(tokenized_data['tokens'], model=model)
        
        # Setup neurons
        layer = model.cfg.n_layers - 1
        if args.neuron_range:
            start, end = map(int, args.neuron_range.split('-'))
            indices = list(range(start, end))
        else:
            indices = list(range(model.cfg.d_mlp))
            
        neurons = [f"{layer}.{i}" for i in indices]
        if args.dry_run:
            neurons = neurons[:10]
        
        # Load unigram distribution
        if 'pythia' in args.model:
            unigram_distrib = get_pile_unigram_distribution(
                device=args.device, file_path='../datasets/pythia-unigrams.npy'
            )
        elif 'gpt' in args.model:
            unigram_distrib = get_pile_unigram_distribution(
                device=args.device, 
                file_path='../datasets/gpt2-small-unigrams_openwebtext-2M_rows_500000.npy',
                pad_to_match_W_U=False
            )
        else:
            raise Exception(f'No unigram distribution for {args.model}')
        
        # Get entropy and activation data
        entropy_df = get_entropy_activation_df(
            neurons,
            tokenized_data,
            token_df,
            model,
            batch_size=min(4, args.batch_size),  # Reduced batch size
            device=args.device,
            cache_residuals=False,
            cache_pre_activations=False,
            compute_kl_from_bu=False,
            residuals_layer=layer,
            residuals_dict={},
        )
        
        # Run ablation
        model.set_use_attn_result(False)
        results = mean_ablate_components(
            components_to_ablate=neurons,
            tokenized_data=tokenized_data,
            entropy_df=entropy_df,
            model=model,
            k=args.k,
            device=args.device,
            chunk_size=getattr(args, 'chunk_size', 5),  # Default to smaller chunk size
            unigram_distrib=unigram_distrib
        )
        
        # Process results
        final_df = pd.concat(results.values())
        final_df = filter_entropy_activation_df(
            final_df.reset_index(),
            model_name=args.model,
            tokenizer=tokenizer,
            start_pos=3,
            end_pos=-1
        )
        
        # Save results
        final_df = final_df.reset_index(drop=True)
        final_df.to_feather(f'{save_path}/k{args.k}.feather')
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise e
    finally:
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    print(f'current dir: {os.getcwd()}')
    run_and_store_ablation_results()