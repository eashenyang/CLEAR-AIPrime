# AI PRIME Readme for CLEAR Challenge
### Welcome to our readme, first you need to modify dataset path in main.py line 36-37. Training scripts are compatible with multiple GPUs.
## CLEAR 100 Train

### Single 1080Ti Train
```
python main.py
```
### DDP Train
```
python -m torch.distributed.launch --nproc_per_node=8 main.py
```

## CLEAR 10 Train

### Single 1080Ti Train
```
python main.py --clear_dataset clear10
```
### DDP Train
```
python -m torch.distributed.launch --nproc_per_node=8 main.py --clear_dataset clear10
```

# Test

### After training, find models with name=='model_ema{trial}' in exp folder and move them to 'models' folder for testing, correct size of single model would be around 93Mb.
### For CLEAR100, select models trial 1-10 for testing.
### For CLEAR10,  select models trial 0-9 for testing.