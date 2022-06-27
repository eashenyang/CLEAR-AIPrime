# AI PRIME Solution for CLEAR Challenge
Welcome! First you need to modify dataset path in main.py line 36-37. Training scripts are compatible with multiple GPUs.  
No evaluation during training.
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
Our results were tested on 8 x RTX3090, for more details of result, check result folder.  
Performance between single GPU and multiple GPUs might be different.  
After training, find models with name=='model_ema_last_{trial}' or name=='model_last_{trial}' in 'exp' folder and move them to 'models' folder for testing, correct size of single model would be around 93Mb.  
For CLEAR100, select models trial 1-10 for testing.  
For CLEAR10,  select models trial 0-9 for testing.
## CLEAR 100 Test
```
python local_evaluation_clear100.py
```
## CLEAR 10 Test
```
python local_evaluation_clear10.py
```