"""Modello del G1 "per le cadute": uguale a quello della camminata di Playground
(stessi giunti, motori e sensori, quindi la rete addestrata funziona identica),
ma con busto, bacino, testa, braccia e gambe che toccano il pavimento.
Nel modello originale toccano terra solo i piedi: cadendo, il corpo attraversa il suolo.

Uso:  import g1_falls; g1_falls.install()   # PRIMA di registry.load("G1JoystickFlatTerrain")
      g1_falls.install(attrito=0.2)          # pavimento scivoloso (attrito normale ~0.8)
"""
import re

from mujoco_playground._src.locomotion.g1 import g1_constants as consts

XML_DIR = consts.ROOT_PATH / "xmls"
BODY_IN, BODY_OUT = "g1_mjx_feetonly.xml", "g1_mjx_fullfall.xml"
SCENE_IN, SCENE_OUT = "scene_mjx_feetonly_flat_terrain.xml", "scene_mjx_fullfall_flat_terrain.xml"

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


def _attrito(xml, mu):
  """Riscrive l'attrito di tutte le coppie di contatto con il pavimento."""
  def sub(m):
    t = m.group(0)
    if 'floor' not in t:
      return t
    if 'friction=' in t:
      return re.sub(r'friction="[^"]*"', f'friction="{mu} {mu}"', t)
    return t[:-2] + f' friction="{mu} {mu}"/>'
  return re.sub(r'<pair\b[^>]*/>', sub, xml)


def install(attrito=None):
  body = (XML_DIR / BODY_IN).read_text()
  for anchor, geom in NEW_GEOMS.items():
    assert body.count(anchor) == 1, anchor
    i = body.index(anchor)
    body = body[:i] + geom + "\n            " + body[i:]
  pairs = "\n".join(f'    <pair name="{g}_floor" geom1="{g}" geom2="floor" condim="3" friction="0.8 0.8"/>'
                    for g in FLOOR_PAIRS)
  body = body.replace("  </contact>", pairs + "\n  </contact>")
  if attrito is not None:
    body = _attrito(body, attrito)
  (XML_DIR / BODY_OUT).write_text(body)
  scene = (XML_DIR / SCENE_IN).read_text().replace(BODY_IN, BODY_OUT)
  (XML_DIR / SCENE_OUT).write_text(scene)
  consts.task_to_xml = lambda task: XML_DIR / SCENE_OUT
  from mujoco_playground._src.locomotion.g1 import joystick
  joystick.consts.task_to_xml = consts.task_to_xml
  return XML_DIR / SCENE_OUT
