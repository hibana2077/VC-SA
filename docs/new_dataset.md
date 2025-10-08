# New Dataset

## HMDB51 Dataset

### Download the dataset

using git clone

```bash
git clone https://huggingface.co/datasets/divm/hmdb51
```

### Dataset Structure

```
hmdb51/
├── train/
│   ├── metadata.csv
│   ├── stand_TheBoondockSaints_stand_f_cm_np1_fr_med_59.mp4
│   ├── stand_TheBoondockSaints_stand_u_cm_np1_fr_bad_45.mp4
│   └── ...
├── validation/
│   ├── metadata.csv
│   ├── walk_Crash_walk_f_cm_np1_le_med_7.mp4
│   ├── walk_Crash_walk_u_nm_np1_ba_med_3.mp4
│   └── ...
├── test/
│   ├── metadata.csv
│   ├── smoke_smoking_smoke_h_nm_np1_fr_med_0.mp4
│   ├── smoke_you_like_a_nice_long_white_cig_smoke_u_nm_np1_ri_goo_2.mp4
│   └── ...
```

### Metadata Format

#### Train/metadata.csv

```
video_id,file_name,label,original_filename,duration,fps,frame_count,width,height,resolution,file_size_original,file_size_mp4,split
eat_Crash_eat_h_cm_np1_fr_med_11,eat_Crash_eat_h_cm_np1_fr_med_11.mp4,eat,Crash_eat_h_cm_np1_fr_med_11.avi,2.6333333333333333,30.0,79,560,240,560x240,266752,167596,train
```

## Breakfast Dataset

### Download the dataset


