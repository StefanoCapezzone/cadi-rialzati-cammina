"""HumanoidGetup: l'umanoide DM Control parte sdraiato a terra e deve rialzarsi.

Variante di mujoco_playground dm_control_suite/humanoid.py (Apache 2.0).
- reset(): il robot parte da una posa "caduta" presa da un archivio di pose
  generate con MuJoCo classico (lancio da sdraiato + assestamento), così
  niente compenetrazioni.
- ricompensa: la ricompensa "stand" originale + un piccolo incentivo ad alzare
  la testa, così l'apprendimento parte anche quando il robot è a terra.
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env, reward
from mujoco_playground._src.dm_control_suite import humanoid as base
from mujoco_playground._src.dm_control_suite import register_environment

STAND_HEIGHT = base._STAND_HEIGHT  # 1.4 m


def default_config() -> config_dict.ConfigDict:
  cfg = base.default_config()
  cfg.episode_length = 600   # 15 s: tempo per rialzarsi e restare in piedi
  cfg.num_fallen_poses = 512
  cfg.progress_weight = 0.3  # peso dell'incentivo "alza la testa"
  return cfg


def generate_fallen_poses(mj_model: mujoco.MjModel, n: int, seed: int = 0,
                          settle_time: float = 2.0) -> np.ndarray:
  """Genera n pose a terra: orientamento a caso (pancia su/giù/fianco),
  giunti a caso nei limiti, lancio da 0,6 m, assestamento senza comandi."""
  rng = np.random.default_rng(seed)
  d = mujoco.MjData(mj_model)
  poses = []
  steps = int(settle_time / mj_model.opt.timestep)
  jnt_range = mj_model.jnt_range[1:]  # tutti i giunti tranne la freejoint
  while len(poses) < n:
    mujoco.mj_resetData(mj_model, d)
    # orientamento: rotazione casuale attorno a un asse orizzontale (sdraiato)
    # composta con un'imbardata casuale
    roll = rng.choice([np.pi / 2, -np.pi / 2, 0.0, np.pi]) + rng.normal(0, .3)
    q_roll = np.zeros(4); mujoco.mju_axisAngle2Quat(q_roll, [1, 0, 0], roll)
    pitch = rng.choice([np.pi / 2, -np.pi / 2]) + rng.normal(0, .3)
    q_pitch = np.zeros(4); mujoco.mju_axisAngle2Quat(q_pitch, [0, 1, 0], pitch)
    yaw = rng.uniform(-np.pi, np.pi)
    q_yaw = np.zeros(4); mujoco.mju_axisAngle2Quat(q_yaw, [0, 0, 1], yaw)
    q = np.zeros(4); mujoco.mju_mulQuat(q, q_yaw, q_pitch)
    if rng.random() < .5:
      mujoco.mju_mulQuat(q, q, q_roll)
    d.qpos[:3] = [0, 0, 0.6]
    d.qpos[3:7] = q
    lo, hi = jnt_range[:, 0], jnt_range[:, 1]
    d.qpos[7:] = rng.uniform(lo, hi) * 0.6
    for _ in range(steps):
      mujoco.mj_step(mj_model, d)
    head_z = d.xpos[mj_model.body("head").id, 2]
    if np.all(np.isfinite(d.qpos)) and head_z < 0.5:  # davvero a terra
      qpos = d.qpos.copy(); qpos[:2] = 0.0             # centrato
      poses.append(qpos)
  return np.stack(poses)


class HumanoidGetup(base.Humanoid):
  """Umanoide che parte a terra: cadi, rialzati."""

  def __init__(self, config: config_dict.ConfigDict = default_config(),
               config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
               fallen_poses: Optional[np.ndarray] = None):
    super().__init__(move_speed=0.0, config=config,
                     config_overrides=config_overrides)
    if fallen_poses is None:
      fallen_poses = generate_fallen_poses(self.mj_model,
                                           self._config.num_fallen_poses)
    self._fallen_poses = jp.array(fallen_poses)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, key = jax.random.split(rng)
    idx = jax.random.randint(key, (), 0, self._fallen_poses.shape[0])
    data = mjx_env.make_data(
        self.mj_model, qpos=self._fallen_poses[idx],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax, njmax=self._config.njmax)
    data = mjx.forward(self.mjx_model, data)
    metrics = {k: jp.zeros(()) for k in (
        "reward/standing", "reward/upright", "reward/stand",
        "reward/small_control", "reward/move", "reward/progress",
        "head_height")}
    info = {"rng": rng}
    obs = self._get_obs(data, info)
    reward_, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward_, done, metrics, info)

  def _get_reward(self, data, action, info, metrics):
    stand_part = super()._get_reward(data, action, info, metrics)
    head_h = self._head_height(data)
    progress = jp.clip(head_h / STAND_HEIGHT, 0.0, 1.0)
    metrics["reward/progress"] = progress
    metrics["head_height"] = head_h
    w = self._config.progress_weight
    return (1 - w) * stand_part + w * progress * metrics["reward/small_control"]


def register():
  """Registra 'HumanoidGetup' in Playground (registry.load lo trova)."""
  register_environment("HumanoidGetup", HumanoidGetup, default_config)
