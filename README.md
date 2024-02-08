# pl-fast

## Steps:
1. Go to `./src/base_config.py` and check every variable. Change if needed.
2. Search file names for "dummy" and change if needed.
3. Global search for "dummy" and change if needed.

## CMDS:
1. `srun --mem=40000 --time=24:00:00 --gres=gpu:1 --pty --constraint=rtx_8000 bash`


## Sweep on SLURM
1. Edit `sweep.yaml` file accordingly: [doc](https://docs.wandb.ai/guides/sweeps)
2. Run `wandb sweep sweep.yaml`
3. Get the sweep ID
4. Run `./sweep.sh $CNT $ID` where `$CNT` is the number of runs and `$ID` is the sweep ID
5. Don't change anything while sweep is running, otherwise it will be run on different code.

# Data Monitoring (DVC)
### Init
```bash
dvc init

dvc remote add --default $REMOTE_NAME gdrive://$ID

dvc remote modify $REMOTE_NAME gdrive_acknowledge_abuse true
```

### Add oauth2 credentials
```bash
dvc remote modify $REMOTE_NAME --local \
      gdrive_user_credentials_file ~/dvc/filename.json
```

### Push Data
```bash
dvc push

dvc add $DATA_FOLDER

dvc push
```

### Stop tracking
```bash
dvc remove $DATA_FOLDER
```