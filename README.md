# CS231N_CZT
Welcome to CS231N project playground!

## Kaggle challenge
* https://www.kaggle.com/c/imet-2019-fgvc6/overview
* https://github.com/visipedia/imet-fgvcx

## Data  

* saved at /mnt/disks/large/data

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

Then you can launch the notebook in browser at http://34.83.253.121:8888 with password cs231n 
