# autonomous_vehicle
![selfdriving](img/self_drive.gif)

The goal of the project is to build a platform that will simulate an autonomous vehicle. The platform should provide essential functionality:
- controlling the vehicle
- collecting the data
- autonomous ride

The following image shows the concept of the simulation platform:
![platform_concept](img/platform_concept.png)
- Prius block - a vehicle with access to steering inputs, odometry and images from cameras
- Joy/Keyboard block - Controlling the vehicle, changing vehicle ride mode (manual/autonomous), turning on/off collecting the data
- Visualization block - Visualizing current velocity, drive mode, steering inputs on front camera image
- Dataset block - collecting the images and labels used to train the CNN model.
- Convolutional Neural Network Model - The trained model that returns the predicted steering angle and vehicle based on input image. The vehicle should ride autonomously after turing on corresponding mode
- PID - the predicted vehicle speed needs to be converted to throttle/brake inputs
- We actually added one more PID block, so the predicted steering angle is also used by PID

## Dependencies
- Ubuntu 18.04 or Ubuntu 20.04
- Docker

## Network architectures
During implementation, we've tested 2 different network architectures. 
- [PilotNet](https://arxiv.org/pdf/2010.08776.pdf?fbclid=IwAR0gxHvJrJUz59P3dSA-cJEfDyx9VJt9h9UNq9GXf9hb74VNLjDN8VhLfjs)
- [Network by M. V. Smolyakov and his team](https://www.researchgate.net/publication/334080652_Self-Driving_Car_Steering_Angle_Prediction_Based_On_Deep_Neural_Network_An_Example_Of_CarND_Udacity_Simulator)

The second network seemed to work better for us. We adapted it a bit and changed the input shape of the image to 800x264
as well as the output shape as we had to predict steering angle and velocity. Final network architecture looks following:
![network_architecture](img/model.png)

## First run
The commands run in host are marked as **H** and the commands from terminal are marked as **C**

- Clone the repository
- Go to the `docker` directory - `H$ cd docker`
- Build the docker image - `H$ ./build.sh` 
- Run the container using `H$ run_cpu.sh` or `H$ run_gpu.sh`
- Go to main workspace `C$ cd /av_ws`
- Initialize and build the workspace (It might take long) `C$ catkin init`, `C$ catkin build`
- Load environment variables `C$ source/av_ws/devel/setup.bash`
- Run demo package `C$ roslaunch car_demo demo.launch`
- [Save the docker container](#saving-docker-container)
- Close the container 
- Create workspace on your local machine `H$ mkdir -p ~/av_ws/src`
- Move the `av_03` and `av_msgs` folder to `H$ ~/av_ws/src` directory
- Make sure that `--volume` arguments in `docker/run_gpu.sh` or `docker/run_cpu.sh` points to the correct directories containing `av_03` and `av_msgs`
    * Sometimes you need to change $USER value to your real username
- Run the container `H$ run_cpu.sh` or `H$ run_gpu.sh`
- Go to the main workspace directory and then to `av_03` package and see if the files are there
```
C$ cd /av_ws/src/av_03/
C$ ls
```

## Running the package
- Run the docker container
- Go to the workspace directory
- Download the trained model and put it in `cnn_models` in `av_03` package 
- Build catkin package
```
C$ catkin build
```
- Source the environment
```
C$ source devel/setup.bash
```
- Launch the demo
```
C$ roslaunch av_03 av.launch
```

### Selfdriving
To start the selfdriving run:
```
C$ rostopic pub --once /prius/mode av_msgs/Mode "{header:{seq: 0, stamp:{secs: 0, nsecs: 0}, frame_id: ''}, selfdriving: true, collect: false}"
```

### Collecting the data 
To collect the data you need to launch controller_node. To do so uncomment line 31 in `av.launch` file.
Then pressing `C` will start collecting data. The steering is being done by the arrow keys

## Saving docker container
```
H$ docker container ps
H$ docker commit container_name av:master
```