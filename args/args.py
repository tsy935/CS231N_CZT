import argparse

def get_args():
    parser = argparse.ArgumentParser('Train a ResNet50 on iMet')
    
    parser.add_argument('--save_dir',
                        type=str,
                        default=None,
                        help='Directory to save the outputs and checkpoints.')
    parser.add_argument('--do_train',
                        default=True,
                        action='store_true',
                        help='To train the model.')
    parser.add_argument('--use_pretrained',
                        default=True,
                        action='store_true',
                        help='Whether to use pre-trained model.')
    parser.add_argument('--model_name',
                        type=str,
                        default='baseline',
                        choices=('baseline','cnn-rnn'),
                        help='Which model to use.')
    parser.add_argument('--lr_init',
                        type=float,
                        default='1e-3',
                        help='Initial learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0.0,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs for which to train.')
    parser.add_argument('--feature_extracting',
                        default=False,
                        action='store_true',
                        help='Use pre-trained model as feacture extracting only or not (fine-tune).')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F2',
                        choices=('F2', 'F1', 'loss'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--eval_steps',
                        type=int,
                        default=10000,
                        help='Number of steps between successive evaluations.')   
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=32,
                        help='Training batch size.')    
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--transform_name',
                        type=str,
                        default='resize_crop',
                        choices=('resize_only','crop_only','resize_crop'),
                        help='Data preprocessing step to be used for evaluation.')
    parser.add_argument('--max_pos_weight',
                        type=float,
                        default=100.,
                        help='Maximum value for pos_weights to weight the imbalanced classes in loss function.')
    parser.add_argument('--loss_fn_name',
                        type=str,
                        default='BCE',
                        choices=('BCE','focal'),
                        help='Name of loss function to be used for training.')
    parser.add_argument('--gamma',
                        type=float,
                        default=2.0,
                        help='Gamma hyperparameter for focal loss.')    
    parser.add_argument('--nlayers_to_freeze',
                        type=int,
                        default=None,
                        help='Number of early layers to freeze.')
    parser.add_argument('--do_predict',
                        default=True,
                        action='store_true',
                        help='To evaluate the model.')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='Dev/test batch size.')
    parser.add_argument('--split',
                        type=str,
                        default='test',
                        choices=('train','dev','test'),
                        help='Split used for testing. Prediction results will be written to csv file.') 
    parser.add_argument('--best_val_results',
                        type=str,
                        default=None,
                        help='Saved best validation set results.')
    parser.add_argument('--metric_avg',
                        type=str,
                        default='samples',
                        help='Averaing method to compute evaluation metrics.')
    parser.add_argument('--write_outputs',
                        default=True,
                        action='store_true',
                        help='Whether write prediction to a csv file.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--extract_hog_feature',
                        default=False,
                        help='Whether to extract hog features while loading data')
    
    args = parser.parse_args()  
    
    
    if args.metric_name == 'loss':
        # Best checkpoint is the one that minimizes loss
        args.maximize_metric = False
    elif args.metric_name in ('F2', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError('Unrecognized metric name: "{}"'.format(args.metric_name))
        
    if (not args.do_train) and (not args.do_predict):
        raise ValueError('At least one of do_train or do_predict must be true.')
    
    if (not args.do_train) and (args.load_path is None or args.best_val_results is None):
        raise ValueError('For prediction only, please provide path to trained model and best_val_results.')
        
  
    return args        
    
    
    
    
