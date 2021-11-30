Lesion Detection

Train in multi gpu
bash ./tools/dist_train.sh configs/lesion/faster_rcnn_r50_fpn_1x_coco.py 2

Inference in mutli gpu
bash ./tools/dist_test.sh configs/lesion/faster_rcnn_r50_fpn_1x_coco.py ./work_dirs/baseline_faster-rcnn/epoch_7.pth 2 --format-only --eval-options "jsonfile_prefix=./work_dirs/baseline_faster-rcnn/baseline_fater-rcnn"