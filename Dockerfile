# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Start FROM Ubuntu image https://hub.docker.com/_/ubuntu
FROM ultralytics/yolov5:latest-cpu

RUN pip3 install git

ENTRYPOINT [ "bash" ]
# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest-cpu && sudo docker build -f utils/docker/Dockerfile-cpu -t $t . && sudo docker push $t

# Pull and Run
# t=ultralytics/yolov5:latest-cpu && sudo docker pull $t && sudo docker run -it --ipc=host -v "$(pwd)"/datasets:/usr/src/datasets $t
