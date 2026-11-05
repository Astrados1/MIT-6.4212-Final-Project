import json,math
from pathlib import Path
from pxr import Usd,UsdPhysics
stage=Usd.Stage.Open('/opt/tiptop-project/droid-sim-evals/assets/franka_robotiq_2f_85_flattened.usd')
rows=[]
for prim in stage.Traverse():
 if prim.IsA(UsdPhysics.RevoluteJoint):
  j=UsdPhysics.RevoluteJoint(prim)
  rows.append({'name':prim.GetName(),'path':str(prim.GetPath()),'lower_rad':math.radians(j.GetLowerLimitAttr().Get()),'upper_rad':math.radians(j.GetUpperLimitAttr().Get())})
Path('/workspace/simulator-joint-limits.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
