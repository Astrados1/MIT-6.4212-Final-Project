# Stage report: arm

Written 20261106T015743Z. Result: **OK**.

| check | status | detail |
|---|---|---|
| arm:info | PASS | largest end-of-move tracking error 0.0002 rad over 28 moves |
| arm:info | PASS | joint 6 commanded 4.219 rad stopped at 3.7525 rad, the simulator limit 3.7525; the planner's FR3 model allows 4.5169. This is the arm mismatch: FR3 plans that used this range were not executed as planned. Planning for panda_robotiq keeps plans inside it |
| arm:info | PASS | joint 7 commanded 3.000 rad stopped at 2.8973 rad, the simulator limit 2.8973; the planner's FR3 model allows 3.0159. This is the arm mismatch: FR3 plans that used this range were not executed as planned. Planning for panda_robotiq keeps plans inside it |
| arm:info | PASS | gripper observation 0.000 open, 1.000 closed |
| arm:info | PASS | finger joint closure 0.00 open, 1.00 closed (1.0 = pi/4 rad, unvalidated scale) |
| arm | PASS | 30 moves in 108.87 s simulated; watch /workspace/MIT-6.4212-Final-Project/results/stages/arm_second/arm_check.mp4 |
