## build docker image 
```bash
IMG=open-mmlab:mmdetection_pytorch1.9.0_cuda11.1_cudnn8
docker build -t $IMG .
```

## run a docker container
```bash
docker run -it \
--privileged \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
-e DISPLAY=unix${DISPLAY} \
--gpus=all \
--name mmdet \
--shm-size 11100M \
-v /mnt/xt/8T/:/mnt/xt/8T \
$IMG bash
```

```bash
# in the container, install mmdet3d and deps
cd /mnt/xt/8T/CODES/CV/open-mmlab/mmdetection
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pip
pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cityscapesscripts
#pip install -i https://pypi.tuna.tsinghua.edu.cn/simple seaborn
```

## train fcos
```bash
# download cityscape dataset, and extract, and create a soft link under data/

# prepare data
python tools/dataset_converters/cityscapes.py \
    ./data/cityscapes \
    --nproc 8 \
    --out-dir ./data/cityscapes/annotations


# train fcos
python tools/train.py fcos_r34_fpn_gn-head_1xb4_1x_cityscapes.py --auto-scale-lr

# vis loss
#infile=work_dirs/fcos3d_r34-caffe_fpn_head-gn_1xb4-1x_nus-mono3d/20230731_044039/vis_data/20230731_044039.json
#python tools/analysis_tools/analyze_logs.py plot_curve ${infile} --keys loss --out ${infile/.json/_loss.png}

# test 
# python tools/test.py  ./configs/fcos3d/fcos3d_r34_v1_fpn_head-gn_1xb4-1x_nus-mono3d.py ./work_dirs/fcos3d_r34_v1_fpn_head-gn_1xb4-1x_nus-mono3d/epoch_12.pth  --task mono_det --show-dir show_dir
```

## experiments

- general  
  Note *: metric is coco


| No. | Model | Dataset    | config                                     | hardware | mem usage | train time      | bbox_mAP* | bbox_mAP_50 | bbox_mAP_75 | bbox_mAP_s | bbox_mAP_m | bbox_mAP_l |
| --- | ----- | ---------- | ------------------------------------------ | -------- | --------- | --------------- | --------- | ----------- | ----------- | ---------- | ---------- | ---------- |
| 1   | fcos  | cityscapes | fcos_r34_fpn_gn-head_1xb4_1x_cityscapes.py | 1080Ti   | 6.452G    | 20230809_095605 | 0.3060    | 0.5300      | 0.2980      | 0.1160     | 0.2970     | 0.4660     |


- mem usage vs. batch_size
  - model: fcos
  - dataset: cityscapes
  - backbone: resnet34
  - neck: fpn(out_channel=256)

 | head                       | batch size | workers | mem usage(G)  |
 | -------------------------- | ---------- | ------- | ------------- |
 | FCOSHead(feat_channel=256) | 4          | 1       | 6.451         |
 | FCOSHead(feat_channel=256) | 6          | 1       | 9.177         |
 | FCOSHead(feat_channel=256) | 8          | 1       | OOM           |
 | FCOSHead(feat_channel=192) | 4          | 1       | 5.595         |
 | FCOSHead(feat_channel=192) | 6          | 1       | 7.979         |
 | FCOSHead(feat_channel=192) | 8          | 1       | OOM           |
 | FCOSHead(feat_channel=160) | 4          | 1       | 7.993         |
 | FCOSHead(feat_channel=160) | 6          | 1       | 7.362         |
 | FCOSHead(feat_channel=160) | 8          | 1       | 10.292->9.569 |
 | FCOSHead(feat_channel=128) | 4          | 1       | 8.533         |
 | FCOSHead(feat_channel=128) | 6          | 1       | 6.733~6.615   |
 | FCOSHead(feat_channel=128) | 8          | 1       | 9.074~9.575   |




