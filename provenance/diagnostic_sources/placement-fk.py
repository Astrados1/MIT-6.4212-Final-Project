import numpy as np,torch,json
from pathlib import Path
from scipy.spatial.transform import Rotation
from cutamp.robots.franka_robotiq import get_fr3_robotiq_kinematics_model
z=np.load('/workspace/placement-failure-trace.npz');model=get_fr3_robotiq_kinematics_model()
def poses(q):
 state=model.get_state(torch.tensor(q,device='cuda',dtype=torch.float32));p=state.ee_position.cpu().numpy();quat=state.ee_quaternion.cpu().numpy();return p,Rotation.from_quat(quat[:,[1,2,3,0]])
actual,ra=poses(z['joint_pos']);command,rc=poses(z['joint_cmd']);obj=z['objects_center'][:,list(z['object_names']).index('rubiks_cube')];expected=command+rc.apply(ra.inv().apply(obj-actual))
rows=[]
for target_t in [17,25,37.2667,38.1333]:
 i=int(np.argmin(abs(z['t']-target_t)))
 rows.append({'t':float(z['t'][i]),'command_q6':float(z['joint_cmd'][i,5]),'measured_q6':float(z['joint_pos'][i,5]),'commanded_tcp':command[i].tolist(),'measured_joint_fk_tcp':actual[i].tolist(),'tcp_tracking_error_m':float(np.linalg.norm(command[i]-actual[i])),'observed_object_center':obj[i].tolist(),'object_center_if_commanded_pose_reached':expected[i].tolist()})
result={'run_id':'002_s1v2_baseline_t0','scope':'FR3 FK at commanded and measured joints; projected object assumes the observed rigid grasp transform would be preserved. This is a diagnostic prediction, not a counterfactual simulation.','samples':rows}
Path('/workspace/placement-fk.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
