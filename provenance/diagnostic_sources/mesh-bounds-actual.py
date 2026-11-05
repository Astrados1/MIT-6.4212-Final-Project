from isaacsim import SimulationApp
app=SimulationApp({"headless":True})
from pxr import Usd, UsdGeom, UsdPhysics
import numpy as np
import json
out={}
for scene in (1,4,5):
 stage=Usd.Stage.Open(f'/opt/tiptop-project/droid-sim-evals/assets/scene{scene}_0.usd')
 out[str(scene)]={}
 for body in stage.Traverse():
  if not body.HasAPI(UsdPhysics.RigidBodyAPI):continue
  if str(body.GetPath()).startswith('/World/robot'):continue
  frame=UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).RemoveScaleShear()
  inv=frame.GetInverse()
  chunks=[]
  for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
   if not p.IsA(UsdGeom.Mesh):continue
   pts=np.array(UsdGeom.Mesh(p).GetPointsAttr().Get())
   if not len(pts):continue
   xf=np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())*inv)
   chunks.append(pts@xf[:3,:3]+xf[3,:3])
  if chunks:
   p=np.concatenate(chunks);bounds=[p.min(axis=0).tolist(),p.max(axis=0).tolist()]
   out[str(scene)][body.GetName()]=bounds
open('/workspace/mesh-bounds-actual.json','w').write(json.dumps(out,indent=2))
print('MESH_BOUNDS_WRITTEN',flush=True)
app.close()
