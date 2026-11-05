import numpy as np,torch,json
from pathlib import Path
from cutamp.robots.franka_robotiq import get_fr3_robotiq_kinematics_model,get_panda_robotiq_kinematics_model
from scipy.spatial.transform import Rotation
z=np.load('/workspace/smoke-trace.npz');q=torch.tensor(z['joint_pos'],device='cuda',dtype=torch.float32)
fr3=get_fr3_robotiq_kinematics_model().get_state(q);panda=get_panda_robotiq_kinematics_model().get_state(q)
p1=fr3.ee_position.cpu().numpy();p2=panda.ee_position.cpu().numpy();r1=fr3.ee_quaternion.cpu().numpy();r2=panda.ee_quaternion.cpu().numpy()
r1=r1.astype(float);r2=r2.astype(float)
r1/=np.linalg.norm(r1,axis=1,keepdims=True);r2/=np.linalg.norm(r2,axis=1,keepdims=True)
angle=2*np.arccos(np.minimum(1,np.abs((r1*r2).sum(axis=1))))*180/np.pi
k=list(z['object_names']).index('rubiks_cube');obj=z['objects_pos'][:,k];cmd=z['gripper_cmd'];held=(cmd>.5)&(obj[:,2]>.14)
rot=Rotation.from_quat(r1[:,[1,2,3,0]])
relative=rot.inv().apply(z['gripper_pos']-p1)
result={'frames':len(q),'fr3_vs_panda_grasp_frame_translation_m':{'min':float(np.linalg.norm(p1-p2,axis=1).min()),'median':float(np.median(np.linalg.norm(p1-p2,axis=1))),'max':float(np.linalg.norm(p1-p2,axis=1).max())},'fr3_vs_panda_orientation_deg_max':float(angle.max()),'fr3_grasp_frame_to_held_cube_distance_m':np.quantile(np.linalg.norm(p1[held]-obj[held],axis=1),[0,.5,1]).tolist(),'finger_reference_minus_fr3_tcp_local_mean':relative.mean(axis=0).tolist(),'finger_reference_minus_fr3_tcp_local_std':relative.std(axis=0).tolist(),'scope':'Planner FR3/Panda forward kinematics at all measured joint positions in the successful smoke; this is not a full direct USD link-by-link FK comparison.'}
Path('/workspace/fk-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
