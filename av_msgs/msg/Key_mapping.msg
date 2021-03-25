# A key mapping for vehicle control
# It can be configured through the output preview.
# $ roscore
# $ rosrun joy joy_node
# $ rostopic echo /joy
# Check output when button/axis pressed (counted from 0) and change button/axis number below.

uint8 BUTTON_GEAR_FORWARD=4
uint8 BUTTON_GEAR_NEUTRAL=1
uint8 BUTTON_GEAR_REVERSE=0

uint8 BUTTON_COLLECTION_ON=6
uint8 BUTTON_COLLECTION_OFF=7

uint8 AXIS_THROTTLE=4
uint8 AXIS_BRAKE=5
uint8 AXIS_STEERING=0

uint8 AXIS_MODE=7

