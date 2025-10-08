# New Dataset

## HMDB51 Dataset

### Download the dataset

using curl

```bash
curl -L -o hmdb51.zip https://huggingface.co/datasets/hibana2077/sample-action-reg-data/resolve/main/hmdb51.zip?download=true
unzip hmdb51.zip -d ./datasets/hmdb51
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

Using curl

```bash
curl -L -o Diving48_rgb.tar.gz https://huggingface.co/datasets/bkprocovid19/diving48/resolve/main/Diving48_rgb.tar.gz?download=true
tar -xzvf Diving48_rgb.tar.gz # output folder rgb
```

Need to using label file from:

- "../src/core/constant/Div48/Diving48_V2_test.json"
- "../src/core/constant/Div48/Diving48_V2_train.json"
- "../src/core/constant/Div48/Diving48_vocab.json"

### Dataset Structure

```
rgb/
├── 2x00lRzlTVQ_00000.mp4
├── 2x00lRzlTVQ_00001.mp4
└── ...
```

### Label Format

#### Diving48_V2_train.json

```json
[
  {
    "vid_name": "-mmq0PT-u8k_00155",
    "label": 0,
    "start_frame": 0,
    "end_frame": 48
  },
  {
    "vid_name": "-mmq0PT-u8k_00156",
    "label": 0,
    "start_frame": 0,
    "end_frame": 70
  },
    ...
]
```

#### Diving48_vocab.json

```json
[
  [
    "Back", 
    "15som", 
    "05Twis", 
    "FREE"
  ], 
  [
    "Back", 
    "15som", 
    "15Twis", 
    "FREE"
  ], 
  [
    "Back", 
    "15som", 
    "25Twis", 
    "FREE"
  ],
    ...
]
```