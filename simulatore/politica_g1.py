"""Rete della camminata del G1 in numpy puro + test senza grafica.

Riproduce G1JoystickFlatTerrain (mujoco_playground 0.2.0) in MuJoCo CPU:
osservazione "state" (103), normalizzazione, MLP 512-256-128 swish, tanh, ctrl = posa + 0.5*azione.

  python politica_g1.py --pesi pesi/step_final.npz --cmd 0.5 0 0 --secondi 10
  python politica_g1.py --pesi pesi/step_final.npz --spenta      # rete spenta: deve cadere
  python politica_g1.py --cadute                                  # modello a collisioni complete
"""
import argparse, os
import numpy as np
import mujoco

QUI = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(QUI, "modello", "scene_mjx_feetonly_flat_terrain.xml")
CTRL_DT, ACTION_SCALE = 0.02, 0.5

# Modello "per le cadute" (come riferimenti/demo/g1_falls.py): stessi giunti, motori e sensori,
# ma bacino, busto, testa, braccia e gambe toccano il pavimento (nell'originale solo i piedi).
# Le inerzie sono esplicite (<inertial>), quindi le geometrie nuove non cambiano le masse.
NEW_GEOMS = {
    '<site name="imu_in_pelvis"': '<geom name="pelvis_col" class="collision" type="sphere" size="0.1" pos="0 0 -0.05"/>',
    '<site name="imu_in_torso"': '<geom name="torso_col" class="collision" type="capsule" size="0.11" fromto="0 0 0.1 0 0 0.36"/>\n'
                                 '            <geom name="head_col" class="collision" type="sphere" size="0.08" pos="0 0 0.44"/>',
    '<joint name="left_shoulder_yaw_joint" class="shoulder_yaw"/>': '<geom name="left_uarm_col" class="collision" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.08"/>',
    '<joint name="right_shoulder_yaw_joint" class="shoulder_yaw"/>': '<geom name="right_uarm_col" class="collision" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.08"/>',
    '<joint name="left_elbow_joint" class="elbow"/>': '<geom name="left_farm_col" class="collision" type="capsule" size="0.035" fromto="0 0 0 0.1 0 0"/>',
    '<joint name="right_elbow_joint" class="elbow"/>': '<geom name="right_farm_col" class="collision" type="capsule" size="0.035" fromto="0 0 0 0.1 0 0"/>',
}
FLOOR_PAIRS = ["pelvis_col", "torso_col", "head_col", "left_thigh", "right_thigh", "left_shin", "right_shin",
               "left_hand_collision", "right_hand_collision", "left_uarm_col", "right_uarm_col",
               "left_farm_col", "right_farm_col"]


def xml_cadute():
  """Scrive modello/scene_fullfall.xml (collisioni complete) e ne restituisce il percorso."""
  d = os.path.join(QUI, "modello")
  body = open(os.path.join(d, "g1_mjx_feetonly.xml")).read()
  for anchor, geom in NEW_GEOMS.items():
    assert body.count(anchor) == 1, anchor
    i = body.index(anchor)
    body = body[:i] + geom + "\n            " + body[i:]
  pairs = "\n".join(f'    <pair name="{g}_floor" geom1="{g}" geom2="floor" condim="3" friction="0.8 0.8"/>'
                    for g in FLOOR_PAIRS)
  body = body.replace("  </contact>", pairs + "\n  </contact>")
  open(os.path.join(d, "g1_mjx_fullfall.xml"), "w").write(body)
  scene = open(XML).read().replace("g1_mjx_feetonly.xml", "g1_mjx_fullfall.xml")
  out = os.path.join(d, "scene_fullfall.xml")
  open(out, "w").write(scene)
  return out


class Rete:
  def __init__(self, npz):
    z = np.load(npz)
    self.mean = z["[0].mean['state']"].astype(np.float64)
    self.std = z["[0].std['state']"].astype(np.float64)
    self.W = [z[f"[1]['params']['hidden_{i}']['kernel']"].astype(np.float64) for i in range(4)]
    self.b = [z[f"[1]['params']['hidden_{i}']['bias']"].astype(np.float64) for i in range(4)]

  def __call__(self, obs):
    x = (obs - self.mean) / self.std
    for i in range(3):
      x = x @ self.W[i] + self.b[i]
      x = x / (1.0 + np.exp(-x))          # swish
    x = x @ self.W[3] + self.b[3]
    return np.tanh(x[:29])                # azione deterministica


class G1:
  def __init__(self, xml=XML, gait_freq=1.375, cadute=False):
    if cadute:
      xml = xml_cadute()
    self.m = mujoco.MjModel.from_xml_path(xml)
    self.d = mujoco.MjData(self.m)
    self.nsub = int(round(CTRL_DT / self.m.opt.timestep))
    k = self.m.key("knees_bent")
    self.q0, self.default_pose = k.qpos.copy(), k.qpos[7:].copy()
    s = lambda n: (self.m.sensor_adr[self.m.sensor(n).id], self.m.sensor_dim[self.m.sensor(n).id])
    self.s_linvel, self.s_gyro = s("local_linvel_pelvis"), s("gyro_pelvis")
    self.imu = self.m.site("imu_in_pelvis").id
    self.phase_dt = 2 * np.pi * CTRL_DT * gait_freq
    self.reset()

  def _sens(self, s):
    return self.d.sensordata[s[0]:s[0] + s[1]].copy()

  def reset(self):
    mujoco.mj_resetData(self.m, self.d)
    self.d.qpos[:] = self.q0
    self.d.ctrl[:] = self.default_pose
    mujoco.mj_forward(self.m, self.d)
    self.last_act = np.zeros(29)
    self.phase = np.array([0.0, np.pi])
    self.cmd = np.zeros(3)

  def obs(self):
    grav = self.d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1.0])
    ph = np.concatenate([np.cos(self.phase), np.sin(self.phase)])
    return np.concatenate([self._sens(self.s_linvel), self._sens(self.s_gyro), grav, self.cmd,
                           self.d.qpos[7:] - self.default_pose, self.d.qvel[6:],
                           self.last_act, ph])

  def step(self, rete=None):
    """Un passo di controllo (20 ms). rete=None: motori fermi nella posa di partenza."""
    if rete is None:
      act = np.zeros(29)
    else:
      act = rete(self.obs())
    self.d.ctrl[:] = self.default_pose + ACTION_SCALE * act
    for _ in range(self.nsub):
      mujoco.mj_step(self.m, self.d)
    self.last_act = act
    self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2 * np.pi) - np.pi


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--pesi", default=os.path.join(QUI, "pesi", "step_final.npz"))
  p.add_argument("--cmd", nargs=3, type=float, default=[0.5, 0, 0])
  p.add_argument("--secondi", type=float, default=10)
  p.add_argument("--spenta", action="store_true")
  p.add_argument("--cadute", action="store_true", help="modello a collisioni complete")
  p.add_argument("--iter", type=int, nargs=2, default=None, help="iterations ls_iterations")
  a = p.parse_args()
  g = G1(cadute=a.cadute)
  if a.iter:
    g.m.opt.iterations, g.m.opt.ls_iterations = a.iter
  rete = None if a.spenta else Rete(a.pesi)
  g.cmd = np.array(a.cmd)
  n = int(a.secondi / CTRL_DT)
  for i in range(n):
    g.step(rete)
    if i % 50 == 0 or i == n - 1:
      print(f"t={i*CTRL_DT:5.1f}s  bacino z={g.d.qpos[2]:.3f}  x={g.d.qpos[0]:6.2f}  y={g.d.qpos[1]:6.2f}")
