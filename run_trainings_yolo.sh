#!/bin/bash

set -e

yolo train data=${DATA_HOME}/trainings_dataset/data.yml project=paper_1 model=yolov8n.pt epochs=250 imgsz=2048 save=True batch=-1 single_cls=True
