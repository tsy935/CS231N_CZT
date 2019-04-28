import numpy as np
import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import utils.utils

from data.dataset import IMetDataset
from args.args import get_args
from collections import OrderedDict
from json import dumps
from models.models import ResNet50
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from constants.constants import TRAIN_PATH, TRAIN_CSV, DEV_PATH, DEV_CSV, TEST_PATH, TEST_CSV, SEED


def main(args):
    # If prediction only, must provide model checkpoint file to load
    if (not args.do_train) and args.do_predict and (args.load_path is None):
        raise ValueError('For prediction only, please provide trained model checkpoint file.')
        
    # Get device
    args.device, args.gpu_ids = utils.get_available_devices()
    
    # Set random seed
    utils.seed_torch(seed=SEED)
    
    # Train
    if args.do_train:
        best_path = train(args)
        
    # Predict
    # TODO: add option to use other models
    if args.do_predict:
        if args.model_name == 'baseline':
            model = ResNet50(args)
#        else:
#            model = CNN_RNN(args)
            
        model = nn.DataParallel(model, args.gpu_ids)
        if args.do_train and (best_path is not None): # load the newly trained model
            model, _ = utils.load_model(model, best_path, args.gpu_ids)
        else: # load from load_path
            model, _ = utils.load_model(model, args.load_path, args.gpu_ids)
         
        model.to(args.device)
        results = evaluate(model, args, args.device, is_test=False, write_outputs=True)
        
        # Log to console
        results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                for k, v in results.items())
        print('{} prediction results: {}'.format(args.split, results_str))
        
        

def train(args):
    # Set up logging and devices
    args.save_dir = utils.get_save_dir(args.save_dir, args.name, training=True)
    log = utils.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info('Using random seed {}...'.format(args.seed))
    utils.seed_torch(args.seed)

    # Get model
    # TODO: add option to use other models
    log.info('Building model...')
    if args.model_name == 'baseline':
        model = ResNet50(args)
#    else:
#        model = CNN_RNN(args)
        
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info('Loading checkpoint from {}...'.format(args.load_path))
        model, step = utils.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(args.device)
    
    # To train mode
    model.train()

    # Get saver
    saver = utils.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adam(params=model.parameters(), 
                           lr=args.lr_init, weight_decay=args.lr_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Define loss function
    if args.loss_fn_name == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean').to(args.device)
    else:
        loss_fn = utils.FocalLoss(gamma=args.gamma)


    # Get data loader
    log.info('Building dataset...')  
    
    train_dataset = IMetDataset(root_dir=Path(TRAIN_PATH),
                                csv_file=TRAIN_CSV,
                                mode='train')
    train_loader = data.DataLoader(dataset=train_dataset,
                                   shuffle=True,
                                   batch_size=args.train_batch_size,
                                   num_workers=args.num_workers)
    

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for imgs, labels, _, _ in train_loader:
                batch_size, ncrops, C, H, W = imgs.size()                
                
                # Setup for forward
                imgs = imgs.to(args.device)
                labels = labels.to(args.device)
                
                # Zero out optimizer first
                optimizer.zero_grad()
                
                # Forward
                y_pred = model(imgs.view(-1, C, H, W)) # fuse batch size and ncrops
                y_pred = y_pred.view(batch_size, ncrops, -1) # shape (batch_size, ncrops, NUM_CLASSES)
                print('Shape of y_pred:{}'.format(y_pred.size()))
                loss = loss_fn(y_pred, labels)
                loss_val = loss.item()

                # Backward
                loss.backward()
                optimizer.step()
                scheduler.step(step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         Loss=loss_val)
                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info('Evaluating at step {}...'.format(step))
                    eval_results = evaluate(model, 
                                            args,
                                            is_test=False,
                                            write_outputs=False)
                    best_path = saver.save(step, model, eval_results[args.metric_name], args.device)
                    
                    # Back to train mode
                    model.train()

                    # Log to console
                    results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                            for k, v in eval_results.items())
                    log.info('Dev {}'.format(results_str))

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in eval_results.items():
                        tbx.add_scalar('dev/{}'.format(k), v, step)

    return best_path


def evaluate(model, args, is_test=False, write_outputs=False):
    
    # Define dataset
    if args.split == 'dev':
        data_dir = DEV_PATH
        csv_file = DEV_CSV
    elif args.split == 'test':
        data_dir = TEST_PATH
        csv_file = TEST_CSV
    else:
        data_dir = TRAIN_PATH
        csv_file = TRAIN_CSV
        
    dataset = IMetDataset(root_dir=Path(data_dir),
                          csv_file=csv_file,
                          mode='evaluate')
    data_loader = data.DataLoader(dataset=dataset,
                                 shuffle=False,
                                 batch_size=args.test_batch_size,
                                 num_workers=args.num_workers)
    
    nll_meter = utils.AverageMeter()
    
    # loss function
    if args.loss_fn_name == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean').to(args.device)
    else:
        loss_fn = utils.FocalLoss(gamma=args.gamma)
    
    # change to evaluate mode
    model.eval()
    
    y_pred_all = []
    y_true_all = []
    orig_id_all = []
    preproc_all = []
    with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
        # If test, no label; in our project, we may not use it
        if is_test:
            for imgs, _, orig_id, preproc in data_loader:
                batch_size, ncrops, C, H, W = imgs.size() 
                
                # Setup for forward
                imgs = imgs.to(args.device)
                
                # Forward
                y_pred = model(imgs.view(-1, C, H, W)) # fuse batch size and ncrops
                y_pred = y_pred.view(batch_size, ncrops, -1).mean(1) # shape (batch_size, 1, NUM_CLASSES), averaged over crops
                
                y_pred_all.append(y_pred.cpu().numpy())
                orig_id_all.append(orig_id)
#                preproc_all.append(preproc)
                
                # Log info
                progress_bar.update(batch_size)
                
        # Else, train/dev
        else:
            for imgs, labels, orig_id, preproc in data_loader:
                batch_size, ncrops, C, H, W = imgs.size()
                
                # Setup for forward
                imgs = imgs.to(args.device)
                labels = labels.to(args.device)

                # Forward
                y_pred = model(imgs.view(-1, C, H, W))
                y_pred = y_pred.view(batch_size, ncrops, -1).mean(1) # shape (batch_size, 1, NUM_CLASSES), averaged over crops

                loss = loss_fn(y_pred, labels)
                nll_meter.update(loss.item(), batch_size)
                
                y_pred_all.append(y_pred.cpu().numpy())
                y_true_all.append(labels.cpu().numpy())
                orig_id_all.append(orig_id)
#                preproc_all.append(preproc)
                
                # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(NLL=nll_meter.avg)
            
    # Get averaged predictions and evaluation metrics
    if is_test:
        results = {}
        writeout_dict = utils.eval_dict(y_pred_all, y_true_all, args.pred_thresh, args.metric_avg, orig_id_all, preproc_all, is_test)            
    else:
        scores_dict, writeout_dict = utils.eval_dict(y_pred_all, y_true_all, args.pred_thresh, args.metric_avg, orig_id_all, preproc_all, is_test)
        results_list = [('Loss', nll_meter.avg),
                        ('F2', scores_dict['F2']),
                        ('F1', scores_dict['F1']),
                        ('recall', scores_dict['recall']),
                        ('precision', scores_dict['precision'])]
        results = OrderedDict(results_list)
    
    # Write prediction into csv file
    if args.write_outputs:
        df_dict = {}            
        for key, val in writeout_dict.items():
            pred1 = np.argwhere(val == 1.0).reshape(-1).tolist()
            pred_str = " ".join(list(map(str, pred1)))
            df_dict[key] = pred_str
        
        df_out = pd.DataFrame(list(df_dict.items()), columns=['id','attribute_ids'])
        out_file_name = os.path.join(args.save_dir,args.split+'_prediction.csv')
        df_out.to_csv(out_file_name, index=False)
        print('Prediction written to {}!'.format(out_file_name))        
            
    return results



if __name__ == '__main__':
    main(get_args())