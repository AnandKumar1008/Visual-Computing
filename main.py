import os
import random
import argparse
import numpy as np
import torch
from datasets import get_all_dataloaders
from utils import *
from sampler import BatchSampler, OnlineSampler
from tqdm import tqdm
from solvers import TransCLIP_solver, StatA_solver, Dirichlet_solver, ZLaP_solver #, TDA_solver, Tent_solver, DMN_solver : only imported when needed to avoid unecessary dependencies

# ── NEW: import the improved solver ──────────────────────────────────────────
from solvers import StatA_N_solver
# ─────────────────────────────────────────────────────────────────────────────

def get_arguments():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument('--dataset', default='dtd', help='dataset name', type=str)
    parser.add_argument('--root_path', default='./datasets', type=str)
    parser.add_argument('--method', default='StatA', type=str, choices=['StatA', 'TransCLIP', 'Dirichlet', 'ZLaP', 'TDA', 'tent', 'DMN'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--backbone', default='vit_b16', type=str, choices=['rn50', 'rn101', 'vit_b32', 'vit_b16', 'vit_l14'], help="CLIP architecture")
    parser.add_argument('--cache_dir', type = str, default = None, help='where to store visual and textual features if not None')
    parser.add_argument('--load', action='store_true', default=False, help="Load features from cache_dir")

    # Experimental arguments
    parser.add_argument('--n_tasks', type=int, default=1, help="number of tasks to run")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--online', action='store_true', default=False, help='online setting or not')
    parser.add_argument('--num_class_eff', type=int, default=None, help='number of effective classes to sample from per batch')
    parser.add_argument('--num_class_eff_min', type=int, default=None, help='number of effective classes per batch minimum')
    parser.add_argument('--num_class_eff_max', type=int, default=None, help='number of effective classes per batch maximum')
    parser.add_argument('--gamma', type = float, default = 1.0, help = 'Dirichlet parameter used for sampling in the online setting.')
    
    # StatA hyperparameters (with default values in all experiments)
    parser.add_argument('--alpha', type=float, default=1.0, help='anchor weighting hyper-parameter')
    parser.add_argument('--lambda_laplacian', type=float, default=1.0, help='Laplacian weighting hyper-parameter')
    parser.add_argument('--soft_beta', action='store_true', default=False, help='use soft beta computation')

    args = parser.parse_args()
    return args

def get_hp(args, method_name):
    if method_name == 'StatA':
        return StatA_solver, {
            'alpha': args.alpha,
            'lambda_y_hat':1,
            'lambda_laplacian': args.lambda_laplacian,
            'n_neighbors':3,
            'soft_beta': args.soft_beta
        }
    elif method_name == 'TransCLIP':
        return TransCLIP_solver, {'lambda_y_hat':1, 'lambda_laplacian': 1, 'n_neighbors':3}
    elif method_name == 'Dirichlet':
        return Dirichlet_solver, {'T':30}
    elif method_name == 'ZLaP':
        return ZLaP_solver, {'k':5, 'gamma':5.0, 'alpha':0.3, 'scale_sim':False}
    elif method_name == 'TDA':
        from solvers import TDA_solver
        # For TDA, we need to know the number of classes to instantiate
        # so the solver is instantiated later in the function main()
        # We use default parameters for ImageNet from TDA's paper
        return None, None
    elif method_name == 'tent':
        return None, None 
    elif method_name == 'DMN':
        # For DMN, we need to know the number of classes to instantiate
        # so the solver is instantiated later in the function main()
        # We use default parameters for ImageNet from DMN's paper
        return None, None
    else:
        raise NotImplementedError(f"Method {args.method} is not implemented.")


# ── NEW: fixed hyperparameter block for StatA_N ──────────────────────────────
def get_hp_N(args):
    """Return the StatA_N solver and its arguments (mirrors StatA block
    but passes the three improvement flags)."""
    return StatA_N_solver, {
        'alpha': args.alpha,
        'lambda_y_hat': 1,
        'lambda_laplacian': args.lambda_laplacian,
        'n_neighbors': 3,
        'soft_beta': args.soft_beta,
        'adaptive_alpha': True,
        'symmetric_affinity': True,
        'adaptive_tau': True,
    }
# ─────────────────────────────────────────────────────────────────────────────


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    args = get_arguments()
    if args.method in ['TDA', 'DMN', 'tent'] and not(args.online):
        raise ValueError(f'Got method {args.method} which is only supported for the online setting, but got args.online = {args.online}.')
    set_random_seed(args.seed) # for reproducibility
    
    if not args.cache_dir:
        args.cache_dir = os.path.join('./caches', args.dataset)
    os.makedirs(args.cache_dir, exist_ok=True)

    # CLIP model
    backbones = {'rn50': 'RN50',
                 'rn101': 'RN101',
                 'vit_b16': 'ViT-B/16',
                 'vit_b32': 'ViT-B/32',
                 'vit_l14': 'ViT-L/14'}
    clip_model, preprocess = clip.load(backbones[args.backbone])
    clip_model.eval()

    # Prepare dataset
    _, _, test_loader, dataset = get_all_dataloaders(args, preprocess)

    # Load features
    test_features, test_labels, clip_prototypes = get_all_features(args, test_loader, dataset, clip_model)
        
    if not args.method == 'tent':
        clip_model = clip_model.to('cpu')  # unload CLIP model from VRAM
        # for tent we need it on cuda

    acc_tot = 0
    acc_zs_tot = 0

    # ── NEW: accumulators for StatA_N (only used when method == 'StatA') ─────
    run_comparison = (args.method == 'StatA')
    acc_tot_N    = 0
    acc_zs_tot_N = 0
    # ─────────────────────────────────────────────────────────────────────────
    
    
    ##############################
    # Batch Test-Time Adaptation #
    ##############################
    
    solver, method_args = get_hp(args, args.method)

    # ── NEW ──────────────────────────────────────────────────────────────────
    if run_comparison:
        solver_N, method_args_N = get_hp_N(args)
    # ─────────────────────────────────────────────────────────────────────────
    
    if not args.online:
        sampler = BatchSampler(test_features, test_labels, args.batch_size, args.num_class_eff, args.num_class_eff_min, args.num_class_eff_max)
    
        for i in tqdm(range(args.n_tasks)):
    
            indices = sampler.generate_indices()
            if indices == None:
                break
            
            preds_zs, preds = solver(test_features[indices,:], test_labels[indices], clip_prototypes, # visual and textual features  
                                              **method_args)
            
            acc_zs = cls_acc(preds_zs, test_labels[indices])
            acc = cls_acc(preds, test_labels[indices])
            acc_zs_tot += acc_zs
            acc_tot += acc

            # ── NEW: run StatA_N on the exact same indices / batch ────────────
            if run_comparison:
                preds_zs_N, preds_N = solver_N(
                    test_features[indices, :], test_labels[indices],
                    clip_prototypes, **method_args_N
                )
                acc_zs_tot_N += cls_acc(preds_zs_N, test_labels[indices])
                acc_tot_N    += cls_acc(preds_N,    test_labels[indices])
            # ─────────────────────────────────────────────────────────────────
        
        
        acc_zs_tot /= args.n_tasks
        acc_tot /= args.n_tasks

        # ── NEW ──────────────────────────────────────────────────────────────
        if run_comparison:
            acc_zs_tot_N /= args.n_tasks
            acc_tot_N    /= args.n_tasks
        # ─────────────────────────────────────────────────────────────────────
        
    ###############################
    # Online Test-Time Adaptation #
    ###############################

    if args.online:
        if args.method == 'TDA':
            K = torch.max(test_labels)+1
            d = test_features.shape[1]
            from solvers import TDA_solver#, run_test_tda, compute_tda_logits
        elif args.method == 'tent':
            K = torch.max(test_labels)+1
            from solvers import Tent_solver, get_cfg
        elif args.method == 'DMN':
            K = torch.max(test_labels)+1
            d = test_features.shape[1]
            from solvers import DMNClipWrapper, get_cfg_DMN, DMNDualMem, select_confident_samples
            import torchvision.transforms as transform
            dmn_args, beta = get_cfg_DMN()

            
        for i in tqdm(range(args.n_tasks)):
            if args.method == 'TDA':
                solver = TDA_solver(K, d) # reinstantiate solver with empty cache
            elif args.method == 'tent':
                solver = Tent_solver(get_cfg('tent'),  clip_model.visual, K) # reinstantiate from unchanged model for each task
            elif args.method == 'DMN':
                DMN_clip = DMNClipWrapper(clip_model, preprocess, 'cuda', 
                            dataset.classnames, args.batch_size, 
                            arch = backbones[args.backbone],).cuda()
                DMN_clip.reset_classnames(dataset)
                DMN_clip.get_text_features()
                dmn = DMNDualMem(args = dmn_args, feat_dim = test_features.shape[-1], class_num = K) # reinstantiate with empty cache
                dmn = dmn.cuda()
                DMN_clip.eval()
                dmn.eval()

            num_batch = test_features.shape[0]//args.batch_size
            num_slots = min(num_batch, len(torch.unique(test_labels)))
            sampler = OnlineSampler(test_features, test_labels, args.gamma, num_slots, args.batch_size)
            
            indices = sampler.generate_indices()
            all_accs = []
            all_accs_zs = []

            # ── NEW ──────────────────────────────────────────────────────────
            all_accs_N    = []
            all_accs_zs_N = []
            # ─────────────────────────────────────────────────────────────────
            
            while indices is not None:
                if args.method == 'tent':
                    batch_imgs = torch.stack([test_loader.dataset[u][0] for u in indices], dim = 0).cuda()
                    preds = solver(batch_imgs, 
                                            clip_prototypes = clip_prototypes.squeeze().T, # for tent we need to recompute the visual features so we pass the images 
                                            )
                    preds = preds.cpu()
                    preds_zs = (test_features[indices,:].cuda() @ clip_prototypes.squeeze()).cpu() # it's a little dumb but preds_zs is actually the (log) probs
                elif args.method == 'DMN':
                    batch_imgs = torch.stack([test_loader.dataset[u][0] for u in indices], dim = 0).cuda()
                    all_img_global_pred = torch.zeros((len(indices), K), dtype = torch.float16,  device = 'cuda')
                    with torch.autocast("cuda"), torch.no_grad():
                        image_features_global = test_features[indices,...].cuda()
                        text_logits = DMN_clip.logit_scale.exp()*image_features_global @ clip_prototypes.squeeze()
                        text_probs = text_logits.softmax(1)
                        for ju,u in enumerate(indices):
                            DMN_clip.image_features_global = image_features_global[ju:ju+1,...]
                            dmn.init_pred = text_probs[ju:ju+1,:]
                            dmn.update_memory_bank(DMN_clip)
                         # predict on the batch with updated memory
                        with torch.autocast("cuda"), torch.no_grad(): #
                            for ju,u in enumerate(indices):
                                # We also need to loop because get_image_pred is not designed to handle batches
                                DMN_clip.image_features_global = image_features_global[ju:ju+1,...]
                                all_img_global_pred[ju,...] = dmn.get_image_pred(DMN_clip, return_logit = True)  ## with updated local
                        preds = (text_logits + beta * all_img_global_pred).squeeze().cpu()  
                        preds_zs = (test_features[indices,:].cuda() @ clip_prototypes.squeeze()).cpu() # it's a little dumb but preds_zs is actually the (log) probs
                else:
                    preds_zs, preds = solver(test_features[indices,:], test_labels[indices], clip_prototypes, **method_args)

                    # ── NEW: run StatA_N on same online batch ─────────────────
                    if run_comparison:
                        preds_zs_N, preds_N = solver_N(
                            test_features[indices, :], test_labels[indices],
                            clip_prototypes, **method_args_N
                        )
                        all_accs_zs_N.append(cls_acc(preds_zs_N, test_labels[indices]))
                        all_accs_N.append(cls_acc(preds_N,    test_labels[indices]))
                    # ─────────────────────────────────────────────────────────
                    
                acc_zs = cls_acc(preds_zs, test_labels[indices])
                acc = cls_acc(preds, test_labels[indices])
                all_accs.append(acc)
                all_accs_zs.append(acc_zs)
                indices = sampler.generate_indices()
                
            acc_tot += sum(all_accs)/len(all_accs)
            acc_zs_tot += sum(all_accs_zs)/len(all_accs_zs)

            # ── NEW ──────────────────────────────────────────────────────────
            if run_comparison and all_accs_N:
                acc_tot_N    += sum(all_accs_N)    / len(all_accs_N)
                acc_zs_tot_N += sum(all_accs_zs_N) / len(all_accs_zs_N)
            # ─────────────────────────────────────────────────────────────────

        acc_tot /= args.n_tasks
        acc_zs_tot /= args.n_tasks

        # ── NEW ──────────────────────────────────────────────────────────────
        if run_comparison:
            acc_tot_N    /= args.n_tasks
            acc_zs_tot_N /= args.n_tasks
        # ─────────────────────────────────────────────────────────────────────
      
        
    print("\n============================")
    print("      Final Results         ")
    print("============================")
    print(f"Dataset:         {args.dataset}")
    print(f"Method:          {args.method}")
    print(f"Backbone:        {args.backbone}")
    print(f"Number of Tasks: {args.n_tasks}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Online Setting:  {'Yes' if args.online else 'No'}")
      
    if args.online:
        print(f"Dirichlet Gamma: {args.gamma:.2f}")
    else:
        print(f"Effective Classes Min: {args.num_class_eff_min or 'None'}")
        print(f"Effective Classes Max: {args.num_class_eff_max or 'None'}")
      
    print("----------------------------")
    print(f"ZERO-shot Accuracy:        {acc_zs_tot:.4f}")
    print(f"StatA  FINAL Accuracy:     {acc_tot:.4f}")

    # ── NEW: print StatA_N row only when it was actually run ─────────────────
    if run_comparison:
        delta = acc_tot_N - acc_tot
        sign  = "+" if delta >= 0 else ""
        print(f"StatA_N FINAL Accuracy:    {acc_tot_N:.4f}  ({sign}{delta:.4f})")
    # ─────────────────────────────────────────────────────────────────────────

    print("============================\n")



if __name__ == '__main__':
    main()
