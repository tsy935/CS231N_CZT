import numpy as np
import os
import pickle
import torch
import json
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import utils.utils as utils

from data.dataset import IMetDataset
from data.dataset_hog import IMetDataset_HOG
#from data.dataset import IMetDatasetBase
from args.args import get_args
from collections import OrderedDict
from json import dumps
from models.ResNet import ResNet50

from models.CNN_RNN import CNN_RNN

from models.ResNet_HOGFC import ResNet50_HOGFC

from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from path import Path


from constants.constants import TRAIN_PATH, TRAIN_CSV, DEV_PATH, DEV_CSV, TEST_PATH, TEST_CSV, SEED, ARGS_FILE_NAME, NUM_CLASSES


def main(args):
    # If prediction only, must provide model checkpoint file to load
    if (not args.do_train) and args.do_predict and (args.load_path is None):
        raise ValueError('For prediction only, please provide trained model checkpoint file.')
        
    # Get device
    device, args.gpu_ids = utils.get_available_devices()
    args.train_batch_size *= max(1, len(args.gpu_ids))
    args.test_batch_size *= max(1, len(args.gpu_ids))    
    
    
    # Set random seed
    utils.seed_torch(seed=SEED)
    
    # Get save directories
    if args.do_train:
        train_save_dir = utils.get_save_dir(args.save_dir, training=True)
        args.train_save_dir = train_save_dir
    if args.do_predict:
        test_save_dir = utils.get_save_dir(args.save_dir, training=False)
        args.test_save_dir = test_save_dir
    
    # Save args
    if args.do_train:
        args_file = os.path.join(train_save_dir, ARGS_FILE_NAME)
    else:
        args_file = os.path.join(test_save_dir, ARGS_FILE_NAME)
    with open(args_file, 'w') as f:
         json.dump(vars(args), f, indent=4, sort_keys=True)
    
    # Train
    if args.do_train:
        train(args, device, train_save_dir)      
        
    # Predict
    # TODO: add option to use other models
    if args.do_predict:
        if args.model_name == 'baseline':
            model = ResNet50(args)
        elif args.model_name == 'cnn-rnn':
            model = CNN_RNN(args, device=device)
        elif args.model_name == "baseline_hog":
            model = ResNet50_HOGFC(args)
            
        model = nn.DataParallel(model, args.gpu_ids)
        if not args.do_train: # load from saved model
            model, _ = utils.load_model(model, args.load_path, args.gpu_ids)
            with open(args.best_val_results, 'rb') as f:
                val_results = pickle.load(f)
        else: # load from newly trained model
            best_path = os.path.join(train_save_dir, 'best.pth.tar')
            best_val_results = os.path.join(train_save_dir, 'best_val_results')
            model, _ = utils.load_model(model, best_path, args.gpu_ids)
            with open(best_val_results, 'rb') as f:
                val_results = pickle.load(f)
                
        if args.model_name == 'baseline' or args.model_name=='baseline_hog':
            best_thresh = val_results['best_thresh']
        else:
            best_thresh = None
         
        model.to(device)
        results, vis_dict = evaluate(model, args, test_save_dir, 
                                     device, is_test=True, 
                                     write_outputs=True, 
                                     best_thresh=best_thresh)
        
        # Log to console
        results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                for k, v in results.items())
        print('{} prediction results: {}'.format(args.split, results_str))
        
        # Save alphas and results
        test_results_dir = os.path.join(test_save_dir, 'test_results')
        with open(test_results_dir, 'wb') as f:
            pickle.dump(results, f)
            
        if args.model_name == 'cnn-rnn':
            test_alphas_dir = os.path.join(test_save_dir, 'test_visualization')
            with open(test_alphas_dir, 'wb') as f:
                pickle.dump(vis_dict, f) 
            
        

def train(args, device, train_save_dir):
    # Set up logging and devices   
    log = utils.get_logger(train_save_dir, 'train')
    tbx = SummaryWriter(train_save_dir)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    
    # Define loss function
    pos_weights = utils.comp_pos_weights(TRAIN_CSV, args.max_pos_weight)
    if args.loss_fn_name == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(pos_weights, dtype=torch.float)).to(device) # more stable than BCELoss
    else:
        loss_fn = utils.FocalLoss(gamma=args.gamma)

    # Get model
    log.info('Building model...')
    if args.model_name == 'baseline':
        model = ResNet50(args)
    elif args.model_name == 'cnn-rnn':
        model = CNN_RNN(args, device)
    elif args.model_name == "baseline_hog":
        model = ResNet50_HOGFC(args)
        
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info('Loading checkpoint from {}...'.format(args.load_path))
        model, step = utils.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    
    # To train mode
    model.train()

    # Get saver
    saver = utils.CheckpointSaver(train_save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adam(params=model.parameters(), 
                           lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    



    # Get data loader
    log.info('Building dataset...')  
    if args.extract_hog_feature == True or args.model_name=="baseline_hog":
        train_dataset = IMetDataset_HOG(root_dir=Path(TRAIN_PATH),
                                csv_file=TRAIN_CSV,
                                mode='train')
    else:
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
            for imgs, labels, _, _, hogs in train_loader:
                batch_size, ncrops, C, H, W = imgs.size()  
                
                # Setup for forward
                imgs = imgs.to(device)
                labels = labels.to(device)
                

                # Zero out optimizer first
                optimizer.zero_grad()
                
                # Forward
                if args.model_name == 'baseline' or args.model_name == 'baseline_hog':
                    if args.model_name=='baseline':
                        logits = model(imgs.view(-1, C, H, W)) # fuse batch size and ncrops
                    else:
                        hogs = hogs.to(device)
                        logits = model(imgs.view(-1, C, H, W), hogs) # fuse batch size and ncrops
                    logits = logits.view(batch_size, ncrops, -1) # shape (batch_size, ncrops, NUM_CLASSES)
                
                    loss = loss_fn(logits, labels) # we use BCEWithLogitsLoss
                    loss_val = loss.item()
                else:
                    loss, _, _ = model(imgs.view(-1, C, H, W), 
                                       labels=labels.view(-1, NUM_CLASSES), 
                                       loss_fn=loss_fn,
                                       is_eval=False) # fuse batch size and ncrops
                    loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=loss_val,
                                         lr=optimizer.param_groups[0]['lr'])
                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps
                    
                    # Evaluate and save checkpoint
                    log.info('Evaluating at step {}...'.format(step))
                    eval_results, _ = evaluate(model,                                            
                                            args,
                                            train_save_dir,
                                            device,
                                            is_test=False,
                                            write_outputs=False)
                    best_path = saver.save(step, model, eval_results[args.metric_name], device, eval_results)
                    
                    # Back to train mode
                    model.train()

                    # Log to console
                    results_str = ', '.join('{}: {}'.format(k, v)
                                            for k, v in eval_results.items())
                    log.info('Dev {}'.format(results_str))

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in eval_results.items():
                        tbx.add_scalar('dev/{}'.format(k), v, step)
           
        
        # step lr scheduler
        scheduler.step()
        
    return best_path


def evaluate(model, args, test_save_dir, device, is_test=False, write_outputs=False, best_thresh=None, feature_extractor=None):   
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
        
    if args.extract_hog_feature == True or args.model_name=="baseline_hog":
        dataset = IMetDataset_HOG(root_dir=Path(data_dir),
                          csv_file=csv_file,
                          mode='evaluate')
    else:
        dataset = IMetDataset(root_dir=Path(data_dir),
                          csv_file=csv_file,
                          mode='evaluate')
        
    
    data_loader = data.DataLoader(dataset=dataset,
                                 shuffle=False,
                                 batch_size=args.test_batch_size,
                                 num_workers=args.num_workers)
    
    nll_meter = utils.AverageMeter()
    
    # loss function
    pos_weights = utils.comp_pos_weights(csv_file, args.max_pos_weight)
    if args.loss_fn_name == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(pos_weights, dtype=torch.float)).to(device)
    else:
        loss_fn = utils.FocalLoss(gamma=args.gamma)
    
    # change to evaluate mode
    model.eval()
    
    y_pred_all = []
    y_true_all = []
    orig_id_all = []
    preproc_all = []
    with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:

        for imgs, labels, orig_id,_, hogs in data_loader:
            batch_size, ncrops, C, H, W = imgs.size()
            #batch_size, C, H, W = imgs.size()
                
            # Setup for forward
            imgs = imgs.to(device)
            labels = labels.to(device) # (batch_size, ncrosp, NUM_CLASSES)

            # Forward
            if args.model_name == 'baseline' or args.model_name =='baseline_hog': 
                is_hard_label = False # for baseline, y_pred is soft probabilities
                if args.model_name == 'baseline':
                    logits = model(imgs.view(-1, C, H, W))
                else:
                    hogs = hogs.to(device)
                    logits = model(imgs.view(-1, C, H, W), hogs) # fuse batch size and ncrops
                
                logits = logits.view(batch_size, ncrops, -1).mean(1) # shape (batch_size, NUM_CLASSES), averaged over crops
                y_pred = torch.sigmoid(logits)
                y_pred = y_pred.cpu().numpy()
                if labels is not None: # if label is available
                    labels = labels[:,0,:] # all crops labels are the same
                    loss = loss_fn(logits, labels)
                    nll_meter.update(loss.item(), batch_size)
                    y_true_all.append(labels.cpu().numpy())
                alphas = None
            else:
                is_hard_label = True # for CNN-RNN, y_pred is hard labels
                loss, y_pred, alphas = model(imgs.view(-1, C, H, W), 
                                             labels=labels, 
                                             loss_fn=loss_fn,
                                             is_eval=True) # fuse batch size and ncrops
                y_pred_crops = y_pred                
                if labels is not None:
                    labels = labels[:,0,:]
                    # Predicted labels are the union of all crops
                    y_pred = y_pred.reshape(batch_size, ncrops, -1).sum(1)
                    y_pred[y_pred > 1] = 1 # (batch_size, NUM_CLASSES)
                    nll_meter.update(loss.item(), batch_size)
                    y_true_all.append(labels.cpu().numpy())
                else:
                    # Predicted labels are the union of all crops
                    y_pred = y_pred.reshape(batch_size, ncrops, -1).sum(1)
                    y_pred[y_pred > 1] = 1
                
            y_pred_all.append(y_pred)
            orig_id_all.extend(list(orig_id))
                
            # Log info
            progress_bar.update(batch_size)
        
        # Save last batch alphas and images for visualization
        vis_dict = {}
        if args.model_name == 'cnn-rnn':           
            vis_dict['imgs'] = imgs.cpu().numpy()
            vis_dict['alphas'] = alphas  
            vis_dict['labels_pred'] = y_pred_crops
                 
    
    # if label is available
    if labels is not None:
        if args.thresh_search and not(is_hard_label): # only threshold search on validation set
            thresh = None
        else:
            thresh = best_thresh
        scores_dict, writeout_dict, best_thresh = utils.eval_dict(y_pred_all, y_true_all, args.metric_avg, 
                                                     orig_id_all, is_test=False, 
                                                     thresh_search=args.thresh_search, thresh=thresh, 
                                                     is_hard_label=is_hard_label, 
                                                     baseline_thresh_prop_power=args.baseline_thresh_prop_power)
        results_list = [('Loss', nll_meter.avg),
                        ('F2', scores_dict['F2']),
                        ('F1', scores_dict['F1']),
                        ('recall', scores_dict['recall']),
                        ('precision', scores_dict['precision'])]
        if best_thresh is not None:
            results_list.append(('best_thresh', best_thresh)) 
        results = OrderedDict(results_list)
    else: # if label is not available
        writeout_dict = utils.eval_dict(y_pred_all, y_true_all, args.metric_avg, 
                                        orig_id_all, preproc_all, is_test=True, 
                                        thresh_search=False, thresh=best_thresh,
                                        is_hard_label=is_hard_label,
                                        baseline_thresh_prop_power=args.baseline_thresh_prop_power)
        results = {}

    
    # Write prediction into csv file
    if is_test and args.write_outputs:
        df_dict = {}            
        for key, val in writeout_dict.items():
            pred1 = np.argwhere(val == 1.0).reshape(-1).tolist()
            pred_str = " ".join(list(map(str, pred1)))
            df_dict[key] = pred_str
        
        df_out = pd.DataFrame(list(df_dict.items()), columns=['id','attribute_ids'])
        out_file_name = os.path.join(test_save_dir,args.split+'_prediction.csv')
        df_out.to_csv(out_file_name, index=False)
        print('Prediction written to {}!'.format(out_file_name))        
            
    return results, vis_dict



if __name__ == '__main__':
    main(get_args())
