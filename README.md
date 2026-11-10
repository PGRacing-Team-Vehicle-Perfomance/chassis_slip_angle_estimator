# Chassis_slip_angle_estimator
All calculations are based on article "Kinematics-Based Analytical Solution for Wheel Slip Angle Estimation of a RWD Vehicle with Drift" by Ronnapee Chaichaowarata and Witaya Wannasuphoprasiy.
The code estimates the wheel slip angles based on data from IMU (yaw rate, acceleration y and x) as well as front wheel speeds (as the vehicle is RWD) and steering wheel angle. 

Before running the code some CONSTANT VEHICLE PARAMETERS need to be known: MASS_KG, WHEELBASE_M, TRACK_WIDTH_M, CG_TO_FRONT_AXLE_M (L_F), CG_TO_REAR_AXLE_M (L_R).
Steering ratio values are mapped for a specific steering wheel angle (right and left corner distinction is important).

Time between individual samples needs to be known as well.
