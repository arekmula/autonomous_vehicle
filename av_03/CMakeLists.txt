cmake_minimum_required(VERSION 3.0.2)
project(av_03)

find_package(catkin REQUIRED COMPONENTS
	prius_msgs
	av_msgs
)

catkin_package(CATKIN_DEPENDS
        prius_msgs
	av_msgs
)

