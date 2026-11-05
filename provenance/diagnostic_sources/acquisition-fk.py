import numpy as np,torch,json
from pathlib import Path
from scipy.spatial.transform import Rotation
from cutamp.robots.franka_robotiq import get_fr3_robotiq_kinematics_model
z=np.load('/workspace/acquisition-trace.npz');r=json.loads(Path('/workspace/acquisition-record.json').read_text());model=get_fr3_robotiq_kinematics_model()
state=model.get_state(torch.tensor(z['joint_pos'],device='cuda',dtype=torch.float32));p=state.ee_position.cpu().numpy();q=state.ee_quaternion.cpu().numpy();rot=Rotation.from_quat(q[:,[1,2,3,0]])
k=list(z['object_names']).index(r['target_object']);obj=z['objects_center'][:,k];t=z['t'];objq=z['objects_quat'][:,k];objrot=Rotation.from_quat(objq[:,[1,2,3,0]])
first=int(np.searchsorted(t,r['score']['first_close_t']));relative=rot.inv().apply(obj-p);relative_rot=rot.inv()*objrot
predicted=p+rot.apply(np.repeat(relative[first][None,:],len(t),axis=0))
rows=[]
for target_t in [t[first],t[first]+1.4,14,16.5,r['score']['release_t']-.15]:
 i=int(np.argmin(abs(t-target_t)));rows.append({'t':float(t[i]),'observed_object_center':obj[i].tolist(),'rigid_center_prediction_from_first_close':predicted[i].tolist(),'grasp_center_shift_since_first_close_m':float(np.linalg.norm(relative[i]-relative[first])),'grasp_rotation_since_first_close_deg':float((relative_rot[first].inv()*relative_rot[i]).magnitude()*180/np.pi),'joint_tracking_max_abs_rad':float(abs(z['joint_cmd'][i]-z['joint_pos'][i]).max())})
result={'run_id':r['run_id'],'scope':'Actual-joint FK and observed object poses. Rigid-center prediction assumes the first-close object-to-TCP transform stays fixed. This is an observational diagnostic, not a counterfactual execution.','first_close_t':float(t[first]),'bowl_initial_center':r['initial_object_centers']['_24_bowl'],'samples':rows}
Path('/workspace/acquisition-fk.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
