This is a refactored version of Des3_DeShadow. Original Link: https://github.com/jinyeying/DeS3_Deshadow

### Train 
Use the following command to train the Diffusion transformer model. You can customize training parameters by modifying the file `configs/AISTDshadow.yml`.

**Run the command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_aistd.py --exp_name experiment1
```

### Test 
Use the following command to run the evaluation script and generate prediction images. You can customize training parameters by modifying the file `configs/AISTDshadow.yml`.

**Run the command:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --resume path/to/checkpoint
```
