import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # exp
    parser.add_argument("--exp_name", default="run", type=str,
                        help="Experiment name")
    parser.add_argument("--dump_path", default="dump/", type=str,
                        help="Experiment dump path")
    parser.add_argument("--exp_id", default="", type=str,
                        help="Experiment ID")
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--load_path", default=None, type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)

    parser.add_argument("--paras_path",type=str,default=None,help="参数文件路径")

    # dataset
    parser.add_argument("--data_root", default='data', type=str)
    parser.add_argument("--config_path", default='configs', type=str)
    parser.add_argument("--dataset", default='QM9', type=str)

    # Encoder
    parser.add_argument("--emb_dim", default=128, type=int)
    parser.add_argument("--layer", default=4, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--gnn_type", default='gin', type=str, choices=['gcn', 'gin','pna'])
    parser.add_argument("--pooling_type", default='mean', type=str)
    #MLP
    parser.add_argument("--mlp_hidden", default=128, type=int)
    parser.add_argument("--mlp_layer", default=2, type=int)
    parser.add_argument("--Zi", default=64, type=int,help='the dimension of the latent space of P,namely "Zi"')
    # Model Zi mlp_layer mlp_hidden
    parser.add_argument("--pos_w", default=0.0, type=float)#0.95
    parser.add_argument("--mani_w", default=1.0, type=float)#0.085
    
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--sigma_H", default=0.35, type=float)
    parser.add_argument("--sigma_L", default=0.55, type=float)

    parser.add_argument("--step_scale", default=1.0, type=float)
    parser.add_argument("--delta", default=0.3, type=float,help='the threshold of clash action')
    # Training
    parser.add_argument("--lr", default=1.0, type=float)
    #annealing
    parser.add_argument("--eta_min", default=0.01, type=float)

    parser.add_argument("--bs", default=128, type=int)
    parser.add_argument("--epoch", default=200, type=int)

    parser.add_argument("--early_stop", default=False, type=bool)
    parser.add_argument("--patience", default=30, type=int)

    parser.add_argument("--metric", default='CE', type=str,choices=[ 'MAE',  'RMSD','score_alignment',"CE"])

    #visulization
    parser.add_argument("--smiles", default="O[C@H]1C[C@@H]2C=C[C@@]1(O)C2", type=str)
    parser.add_argument("--get_image", default=True, type=bool)

    #optimization
    parser.add_argument('--run_bayesian_optimization', action='store_true', 
                        help='Run Bayesian optimization for hyperparameters')
    parser.add_argument('--optim_trials', type=int, default=50,
                        help='Number of trials for Bayesian optimization')
    parser.add_argument('--log_dir', type=str, default='/home/liyong/data/opt-log',
                        help='Base directory for logs')
    
    #forcefield
    parser.add_argument('--forcefield', type=str, default='mmff')

    #manifold_type
    parser.add_argument('--train_model', type=str, default='teacher',choices=['teacher', 'D1', 'D2','D3'])
    parser.add_argument('--attn_heads', type=int, default=8, help='Number of attention heads in training "P"') 

    #route_type
    parser.add_argument('--route', type=str, default='A',choices=['A','B'])

    #decide "a" and "b". You should set them before owning a "processed data" folder
    parser.add_argument('--min_dist', type=float, default=0.5,help='the minimum distance between two atoms to consider them as clashing')
    parser.add_argument('--spread', type=float, default=1.0)

    parser.add_argument('--topk', type=int, default=8,help='the maximum distance between two atoms to consider them as clashing')
    parser.add_argument("--use_topk",type=bool,default=False)

    parser.add_argument("--mani_threshold", type=float, default=10000,help='the threshold of manifold action')
    parser.add_argument("--refine_start_epoch", type=int, default=1000,help='the epoch to start refining the manifold')



    parser.add_argument("--verbose", type=bool, default=False,help='whether to print the debug information')
    args = parser.parse_args()
    


    return args
