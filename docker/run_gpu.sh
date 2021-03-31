xhost + local:root

XAUTH=/tmp/.docker.xauth
 if [ ! -f $XAUTH ]
 then
     xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
     if [ ! -z "$xauth_list" ]
     then
         echo $xauth_list | xauth -f $XAUTH nmerge -
     else
         touch $XAUTH
     fi
     chmod a+r $XAUTH
 fi


docker run -it \
    --gpus all \
    --env="DISPLAY=:1" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --volume="/home/$USER/av_ws/src/av_03:/av_ws/src/av_03" \
    --volume="/home/$USER/av_ws/src/av_msgs:/av_ws/src/av_msgs" \
    --env="NVIDIA_VISIBLE_DEVICES=all" \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --privileged \
    --network=host \
    -p 2222:2222 \
    av:master

--device=/dev/input/js0:/dev/input/js0 \
--device=/dev/input/js1:/dev/input/js1 \
--device=/dev/input/js2:/dev/input/js2 \
--device=/dev/input/js3:/dev/input/js3 \
