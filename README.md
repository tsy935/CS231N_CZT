# CS231N_CZT
Welcome to CS231N project playground!

## Kaggle challenge
* https://www.kaggle.com/c/imet-2019-fgvc6/overview
* https://github.com/visipedia/imet-fgvcx

## Data  

* saved at `/mnt/disks/large/data`

### Data Split
* Done in `analyses/preprocessing.ipynb`

* Original Training Data that contains labels that appear less than 3 times are dropped 

* Rest of training data are split 8:1:1 into our own train/val/test. The mapping is in `data/train_split.csv` (with one hot encoding). The ID and attributes are saved in original format in (Github folder) `data/train_split_train.csv`, `data/train_split_val.csv`, `data/train_split_test.csv` and also in vm disk data folder `/mnt/disks/large/data/train_split`

* Original data are `/mnt/disks/large/data/train.zip` and `test.zip`

* Our own train/val/test (original .png files) are stored in `/mnt/disks/large/data/train_split`

* Small dataset for debug: `/mnt/disks/large/debug_data`, `train_split_train.csv`, `train_split_val.csv` and `train_split_test.csv` for this small dataset is also stored here.

## Output
* Please save all experimental results in `/mnt/disks/large/output`

## Code
### Environment
* The environment is defined in `./environment.yml` (called imet), and has been setup in VM `cs231n1-vm`. To activate the environment, run the following on terminal:
```
conda activate imet
```
To deactivate, run the following:
```
conda deactivate imet
```

## Housekeeping
* Always work on your own branch. Use the following command to create a new branch:
```
# make sure your master branch is up to date
git pull origin master
# create a new branch
git checkout -b [name_of_your_new_branch]
```
* Commit your changes to your branch:
```
git add .
git commit
git push origin [name_of_your_new_branch]
```
* After you are confident about the changes you have made in your branch and want to merge to master branch, go to Github website to create a `pull request`

## Storage

* Check storage usage 
```
df
``` 

* You should see a 100GB (base disk for VM (Your home directory is on this disk ). if this disk is full we cannot access VM), and a 300GB large disk mounted at /mnt/disks/large. Our raw data is stored at /mnt/disks/large/data.


* Mount should be done automatically every time VM is started. (see https://cloud.google.com/compute/docs/disks/add-persistent-disk)

* We should write all intermediate data or output files to /mnt/disks/large not folders under  ~/.

##  Jupyter

https://github.com/cs231n/gcloud.git has already been installed and setup.

Jupyter notebook has been already setup.

To launch notebook, Run

```
jupyter notebook
```

Then you can launch the notebook in browser at http://34.83.253.121:8888 with password cs231n, http:34.83.253.121:8800 for Siyi, http:34.83.253.121:8899 for Mingkun 

## Tensorboard
During training, some results such as train/dev loss, train LR, val F2, val F1 etc. will be recorded in an event file. You may view these results using Tensorboard. To use Tensorboard, go to the parent folder of our code, i.e. `CS231N_CZT`, then run the following command:
```
tensorboard --logdir [dir-to-the-folder-of-output-event-file] --port [port-number]
```
Next, on your local machine, run the following command:
```
ssh -N -f -L localhost:1234:localhost:[port-number] <user>@<remote>
```
Now, open http://localhost:1234/ in the browswer on your local machine, you should be able to see the Tensorboard.
