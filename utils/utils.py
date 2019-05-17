import time
import random
import os
import logging
import queue
import shutil
import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from contextlib import contextmanager
from collections import defaultdict
from sklearn.metrics import fbeta_score, f1_score, recall_score, precision_score
from constants.constants import NUM_CLASSES,TRAIN_PROPORTION_PATH



@contextmanager
def timer(name="Main", logger=None):
    t0 = time.time()
    yield
    msg = f"[{name}] done in {time.time() - t0} s"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
      

def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.
    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.
    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.
        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_save_dir(base_dir, training, id_max=50):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).
    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.
    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, '{}-{:02d}'.format(subdir, uid))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


def get_available_devices():
    """Get IDs of all available GPUs.
    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device('cuda:{}'.format(gpu_ids[0]))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids
  

def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.
    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.
    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = 'cuda:{}'.format(gpu_ids[0]) if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model


def eval_dict(y_pred, labels, average, orig_id_all, is_test=False, thresh_search=False, thresh=None, is_hard_label=False):
    """Helper function to compute evaluation metrics: F1 & F2 scores
    Args:
        y_pred: Predicted probabilities of all preprocessed images
        labels: Labels of all preprocessed images
        average: "samples", "macro", "micro" etc. for computing F1 and F2 scores
        orig_id_all: List of original ids, order corresponding to y_pred and labels
        is_test: True if no labels are available, only output writeout_dict
        thresh_search: Whether to do threshold search
        thresh: Threshold to be used for binarizing the prediction results
        is_hard_label: Whether y_pre is hard labels or soft probabilities
    """   
    
    if thresh_search and is_hard_label:
        raise ValueError('y_pred are hard labels, cannot do threshold search.')
    
    scores_dict = {}
    writeout_dict = defaultdict(list)
    
    y_pred = np.concatenate(y_pred, axis=0)
    labels = np.concatenate(labels, axis=0)
    

    proportion = np.array(pd.read_csv(TRAIN_PROPORTION_PATH))
    # threshold search
    if not is_hard_label:
        if thresh_search:
            # thresh, _ = threshold_search(y_pred, labels, average)
            thresh, _ = threshold_search_proportional(y_pred, labels, average,proportion)
        else:
            if thresh is None:
                thresh = 0.5
        
        y_pred_labels = []
        y_labels = []
        for idx, orig_id in enumerate(orig_id_all):
            curr_pred = (y_pred[idx] > thresh).astype(int)
            writeout_dict[orig_id] = curr_pred
            y_pred_labels.append(curr_pred)
            y_labels.append(labels[idx])
        
        y_pred_labels = np.asarray(y_pred_labels)
        y_labels = np.asarray(y_labels)
    else:
        y_pred_labels = y_pred
        y_labels = labels
        
    if not is_test:
        scores_dict['F2'] = fbeta_score(y_true=y_labels, y_pred=y_pred_labels, beta=2, average=average)
        scores_dict['F1'] = f1_score(y_true=y_labels, y_pred=y_pred_labels, average=average)
        scores_dict['recall'] = recall_score(y_true=y_labels, y_pred=y_pred_labels, average=average)
        scores_dict['precision'] = precision_score(y_true=y_labels, y_pred=y_pred_labels, average=average)
        return scores_dict, writeout_dict, thresh
    else:
        return writeout_dict

    
def threshold_search(y_pred, y_true, average):
    """
        Adapted from https://www.kaggle.com/hidehisaarai1213/imet-pytorch-starter
    """
    score = []
    candidates = list(np.arange(0, 0.5, 0.01))
    for _, th in enumerate(candidates):
        yp = (y_pred > th).astype(int)
        score.append(fbeta_score(y_true=y_true, y_pred=yp, beta=2, average=average))
    score = np.array(score)
    pm = score.argmax()
    best_th, best_score = candidates[pm], score[pm]
    return best_th, best_score


def threshold_search_proportional(y_pred, y_true, average, proportion):
    """
        threshold search on a constant that works best so best_th*proportion[i]
         of each label i overall is the best

         input: 
            y_pred:
            y_true: one hot encoded
            average:
            proportions:

    """
    proportion/= np.sum(proportion)
    proportion2d= np.repeat(proportion, y_true.shape[0])
    proportion2d=proportion2d.reshape(y_true.shape,order='F')

    score = []
    candidates = list(np.arange(0, 1/np.max(proportion), 0.01))
    for _, th in enumerate(candidates):

        yp = (y_pred > th*proportion2d[y_true.astype(bool)].reshape(-1,1)).astype(int)
        score.append(fbeta_score(y_true=y_true, y_pred=yp, beta=2, average=average))
    score = np.array(score)
    pm = score.argmax()
    best_th, best_score = candidates[pm], score[pm]
    return best_th, best_score


def my_f2(y_true, y_pred, beta=2.):
    """
        Compute F2 score, adpated from https://www.kaggle.com/mathormad/resnet50-v2-keras-focal-loss-mix-up
    """
    EPSILON = 1e-7
    assert y_true.shape[0] == y_pred.shape[0]

    tp = np.sum((y_true == 1) & (y_pred == 1), axis=1)
    tn = np.sum((y_true == 0) & (y_pred == 0), axis=1)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=1)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=1)

    p = tp / (tp + fp + EPSILON)
    r = tp / (tp + fn + EPSILON)

    f2 = (1+beta**2)*p*r / (p*beta**2 + r + 1e-15)

    return np.mean(f2)


def comp_pos_weights(csv_file, max_pos_weight):
    """
        Helper function to compute the positive weights to be used in BCELoss/BCEWithLogitsLoss
    """
    df = pd.read_csv(csv_file, engine='python')
    label_onehot = np.zeros((len(df), NUM_CLASSES), dtype=int)
    for idx, attr_arr in enumerate(df.attribute_ids.str.split(" ").apply(lambda l: list(map(int, l))).values):
        label_onehot[idx, attr_arr] = 1
        
    frequencies = label_onehot.sum(axis=0)
    
    pos_weights = []
    for i, freq in enumerate(frequencies):
        if freq != 0.:
            pos_weights.append((len(df)-freq) / freq) # positive weight = (num of negative) / (num of positive)
        else: # for removed classes, just set pos_weights to 1
            pos_weights.append(1.)
    
    pos_weights = np.asarray(pos_weights)
    pos_weights = np.clip(pos_weights, a_min=None, a_max=max_pos_weight) # clip the maximum
    return pos_weights


def visualize_attn(img, alphas, labels):
    """
    Visualize attention
    Adapted from https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    Args:
        img: image numpy array to be visualized, (C, H, W)
        alphas: alphas (attention) numpy array, (max_label_len, num_pixels)
        labels: predicted/true labels of the image, one-hot encoded, (NUM_CLASSES,)
    """
    labels_num = np.argmax(labels, axis=1)
    labs = labels_num[labels_num != 0]
    
    H, W = img.shape[1], img.shape[2]
    
    for t in range(len(labs)):
        plt.subplot(np.ceil(len(labs) / 5.), 5, t + 1)
        
        plt.text(0, 1, '%s' % (labs[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(img)
        current_alpha = alphas[t,:]
        
        alpha = skimage.transform.resize(current_alpha, [H, W])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()
    
class CheckpointSaver:
    """Class to save and load model checkpoints.
    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.
    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """
    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print('Saver will {}imize {}...'
                    .format('max' if maximize_metric else 'min', metric_name))

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.
        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_val, device, val_results):
        """Save model parameters to disk.
        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir,
                                       'step_{}.pth.tar'.format(step))
        torch.save(ckpt_dict, checkpoint_path)
        self._print('Saved checkpoint: {}'.format(checkpoint_path))
        
        best_path = ''
        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            best_val_results = os.path.join(self.save_dir, 'best_val_results')
            with open(best_val_results,'wb') as f:
                pickle.dump(val_results,f)
            self._print('New best checkpoint at step {}...'.format(step))

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print('Removed checkpoint: {}'.format(worst_ckpt))
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass
            
        return best_path
            

class AverageMeter:
    """Keep track of average values over time.
    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.
        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count
        

class FocalLoss(nn.Module):
    """Focal loss: Borrowed from this kernel:
    https://www.kaggle.com/backaggle/imet-fastai-starter-focal-and-fbeta-loss#Create-learner-with-densenet121-and-FocalLoss
    https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
    """
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)

        return loss.mean()