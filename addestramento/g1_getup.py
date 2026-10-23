"""G1Getup: il G1 parte a terra e deve rimettersi in piedi.

VERSIONE 3 (22/09/2026). Copie delle precedenti: g1_getup_v1.py, g1_getup_v2.py.

Perche' v1 e v2 si fermavano seduti: la rete comandava ogni giunto solo entro
+-0,5 rad dalla posa in piedi (action_scale della camminata). Il ginocchio arrivava
al massimo a 1,17 rad: portare i piedi sotto il bacino (servono ~2 rad) era
fisicamente impossibile, e anche le braccia non potevano spingere sul pavimento.
Novita' della v3:
  1. AZIONE SU TUTTA L'ESCURSIONE: -1 = fine corsa inferiore, 0 = posa in piedi,
     +1 = fine corsa superiore (per ogni giunto, dentro i limiti "morbidi").
  2. LA MANO CHE TIRA SU (idea di HoST, 2025): una forza verticale sul bacino, pari
     a `assist` volte il peso, applicata solo quando il busto e' gia' dritto (da
     sdraiati non aiuta). Si toglie a tappe tra una run e l'altra (0,6 -> 0,3 ->
     0,1 -> 0). Nei video di verifica vale sempre 0.
  3. Ricompensa anche sul tentativo: altezza della testa e velocita' del bacino
     verso l'alto.
  4. Partenze accovacciate vere: bacino calcolato perche' i piedi tocchino terra,
     piedi piatti per costruzione, nessuna inclinazione di lato.
  5. Episodi da 8 s (e, nel lancio, sconto 0,99 invece di 0,97).

E' una sottoclasse dell'ambiente di camminata (g1.joystick.Joystick): stesse
osservazioni (103) e stesso numero di uscite (29). ATTENZIONE: con l'azione su tutta
l'escursione le uscite NON hanno piu' lo stesso significato di quelle della rete che
cammina (servirebbe un "traduttore" per incatenare le due reti).

Nel G1 get_gravity() legge il sensore "upvector": [0,0,+1] in piedi (nel Go1 -1).

Uso:
  import g1_getup; g1_getup.register()
  env = registry.load("G1Getup")
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

import g1_falls

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion import g1 as _g1_pkg  # noqa: F401
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick

NOME = "G1Getup"
UP = jp.array([0.0, 0.0, 1.0])   # il sensore upvector del G1: [0,0,1] = in piedi
ALTEZZA_IN_PIEDI = 0.755         # bacino nella posa "knees_bent" del modello
TESTA = np.array([0.0, 0.0, 0.44])  # centro della testa nel sistema di torso_link (g1_falls)


def default_config() -> config_dict.ConfigDict:
  cfg = g1_joystick.default_config()

  cfg.episode_length = 400          # 8 secondi

  # Niente spinte e niente comandi di velocita' (sample_command ignora command_config).
  cfg.push_config.enable = False
  cfg.lin_vel_x = [0.0, 0.0]
  cfg.lin_vel_y = [0.0, 0.0]
  cfg.ang_vel_yaw = [0.0, 0.0]

  cfg.naconmax = 24 * 8192
  cfg.njmax = 29 * 2 + 24 * 4

  cfg.full_range_action = True      # (1) azione su tutta l'escursione
  cfg.assist = 0.0                  # (2) frazione del peso; la si alza SOLO in addestramento
  cfg.assist_up_min = 0.5           # la mano comincia a tirare da busto inclinato ~60 gradi
  cfg.assist_up_full = 0.9          # ...e tira del tutto da busto quasi dritto

  # Da dove si parte.
  cfg.drop_from_height_prob = 0.55  # a terra
  cfg.crouch_prob = 0.30            # accovacciato (il resto: in piedi)
  cfg.drop_height = 0.45
  cfg.settle_time = 0.6             # assestamento della partenza "a terra"
  cfg.crouch_settle_time = 0.2      # assestamento breve della partenza accovacciata
  cfg.joint_spread = 0.5
  cfg.height_target = 0.68

  cfg.reward_config = config_dict.create(
      scales=config_dict.create(
          orientation=1.0,
          torso_height=2.0,
          head_height=1.0,        # (3)
          up_vel=0.5,             # (3)
          posture=1.0,
          stand_still=1.0,
          action_rate=-0.002,     # azioni ora ~4 volte piu' ampie: scatti contati sull'azione grezza
          torques=-1e-5,
          dof_pos_limits=-0.1,
          dof_acc=-2.5e-7,
          dof_vel=0.0,
      ),
      ori_tol=0.05,
      height_tol=0.05,
  )
  return cfg


class G1Getup(g1_joystick.Joystick):
  """Cadi, rialzati: il G1 torna in piedi da terra."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(task=task, config=config, config_overrides=config_overrides)

  def _post_init(self) -> None:
    super()._post_init()
    self._settle_steps = int(self._config.settle_time / self.sim_dt)
    self._crouch_settle_steps = max(1, int(self._config.crouch_settle_time / self.sim_dt))
    self._z_des = self._config.height_target
    self._mezzo_intervallo = 0.5 * (self._uppers - self._lowers)
    self._pelvis_body_id = self._mj_model.body("pelvis").id
    self._peso = float(np.sum(self._mj_model.body_mass)) * 9.81

    # Riferimenti in piedi calcolati una volta con MuJoCo "classico": quota dei siti
    # dei piedi e della testa nella posa knees_bent.
    d = mujoco.MjData(self._mj_model)
    mujoco.mj_resetDataKeyframe(self._mj_model, d, self._mj_model.keyframe("knees_bent").id)
    mujoco.mj_forward(self._mj_model, d)
    self._piede_z0 = float(np.mean(d.site_xpos[self._feet_site_id, 2]))
    self._testa_z0 = float((d.xpos[self._torso_body_id] + d.xmat[self._torso_body_id].reshape(3, 3) @ TESTA)[2])

    # Azione su tutta l'escursione: sopra lo zero verso il limite alto, sotto verso il basso.
    self._su = jp.maximum(self._soft_uppers - self._default_pose, 0.05)
    self._giu = jp.maximum(self._default_pose - self._soft_lowers, 0.05)

  # --- utilita' ---------------------------------------------------------------

  def _bersagli(self, action: jax.Array) -> jax.Array:
    if self._config.full_range_action:
      return self._default_pose + jp.where(action > 0, action * self._su, action * self._giu)
    return self._default_pose + action * self._config.action_scale

  def _quota_testa(self, data) -> jax.Array:
    R = data.xmat[self._torso_body_id].reshape(3, 3)
    return (data.xpos[self._torso_body_id] + R @ jp.asarray(TESTA))[2]

  # --- da dove si parte ----------------------------------------------------

  def _posa_a_terra(self, rng: jax.Array) -> jax.Array:
    rng, k_or, k_q = jax.random.split(rng, 3)
    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[2].set(self._config.drop_height)
    quat = jax.random.normal(k_or, (4,))
    quat = quat / (jp.linalg.norm(quat) + 1e-6)
    qpos = qpos.at[3:7].set(quat)
    delta = jax.random.uniform(k_q, (29,), minval=-1.0, maxval=1.0)
    giunti = self._default_pose + delta * self._config.joint_spread * self._mezzo_intervallo
    qpos = qpos.at[7:].set(jp.clip(giunti, self._soft_lowers, self._soft_uppers))
    return qpos

  def _posa_accovacciata(self, rng: jax.Array) -> jax.Array:
    """Accovacciato con i piedi piatti per costruzione.
    Ginocchio k, stinco inclinato in avanti di b, coscia di a = k - b, bacino
    inclinato in avanti di `lean`: anca = -a - lean, caviglia = -b.
    L'altezza del bacino si sistema dopo, in reset(), coi piedi a terra."""
    k_gin, k_lean, k_yaw = jax.random.split(rng, 3)
    k = jax.random.uniform(k_gin, (), minval=0.8, maxval=2.3)
    b = jp.minimum(0.45 * k, 0.8)
    a = k - b
    lean = jax.random.uniform(k_lean, (), minval=0.0, maxval=0.5)
    yaw = jax.random.uniform(k_yaw, (), minval=-3.14, maxval=3.14)
    # quaternione: imbardata poi beccheggio (in avanti = rotazione positiva attorno a y)
    cy, sy = jp.cos(yaw / 2), jp.sin(yaw / 2)
    cp, sp = jp.cos(lean / 2), jp.sin(lean / 2)
    quat = jp.array([cy * cp, -sy * sp, cy * sp, sy * cp])
    qpos = self._init_q.at[3:7].set(quat)
    g = qpos[7:]
    for i_anca, i_gin, i_cav in ((0, 3, 4), (6, 9, 10)):   # gamba sinistra, destra
      g = g.at[i_anca].set(-a - lean).at[i_gin].set(k).at[i_cav].set(-b)
    return qpos.at[7:].set(jp.clip(g, self._soft_lowers, self._soft_uppers))

  def _dati(self, qpos, qvel):
    return mjx_env.make_data(
        self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax, njmax=self._config.njmax,
    )

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = super().reset(rng)
    rng, k_scelta, k_posa, k_acc, k_vel = jax.random.split(state.info["rng"], 5)

    # (a) a terra: lasciato cadere e assestato a motori fermi
    q_terra = self._posa_a_terra(k_posa)
    v_terra = jp.zeros(self.mjx_model.nv).at[0:6].set(
        jax.random.uniform(k_vel, (6,), minval=-0.5, maxval=0.5))
    d_terra = mjx_env.step(self.mjx_model, self._dati(q_terra, v_terra), q_terra[7:],
                           self._settle_steps)

    # (b) accovacciato o in piedi: si abbassa il bacino finche' i piedi toccano terra,
    #     poi un assestamento breve con i motori che tengono la posa
    u = jax.random.uniform(k_scelta)
    p_terra = self._config.drop_from_height_prob
    p_acc = self._config.crouch_prob
    q_acc = jp.where(u < p_terra + p_acc, self._posa_accovacciata(k_acc), self._init_q)
    d0 = mjx.forward(self.mjx_model, self._dati(q_acc, jp.zeros(self.mjx_model.nv)))
    piedi = jp.min(d0.site_xpos[self._feet_site_id, 2])
    q_acc = q_acc.at[2].add(self._piede_z0 - piedi)
    d_acc = mjx_env.step(self.mjx_model, self._dati(q_acc, jp.zeros(self.mjx_model.nv)),
                         q_acc[7:], self._crouch_settle_steps)

    data = jax.tree.map(lambda x, y: jp.where(u < p_terra, x, y), d_terra, d_acc)
    data = data.replace(time=0.0)

    info = dict(state.info)
    info["rng"] = rng
    info["command"] = jp.zeros(3)
    info["last_act"] = jp.zeros(self.mjx_model.nu)
    info["last_last_act"] = jp.zeros(self.mjx_model.nu)

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sid]] > 0
        for sid in self._feet_floor_found_sensor
    ])
    obs = self._get_obs(data, info, contact)
    return state.replace(data=data, obs=obs, info=info)

  # --- un passo: azione su tutta l'escursione + la mano che tira su ------------

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    data = state.data
    su = self.get_gravity(data, "torso")[2]
    presa = jp.clip((su - self._config.assist_up_min)
                    / (self._config.assist_up_full - self._config.assist_up_min), 0.0, 1.0)
    forza = self._config.assist * self._peso * presa
    xfrc = jp.zeros_like(data.xfrc_applied).at[self._pelvis_body_id, 2].set(forza)
    data = data.replace(xfrc_applied=xfrc)

    motor_targets = self._bersagli(action)
    data = mjx_env.step(self.mjx_model, data, motor_targets, self.n_substeps)
    state.info["motor_targets"] = motor_targets

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sid]] > 0
        for sid in self._feet_floor_found_sensor
    ])
    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info, state.metrics, done, None, contact)
    rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
    reward = sum(rewards.values()) * self.dt

    state.info["step"] += 1
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_contact"] = contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.zeros(())

    done = done.astype(reward.dtype)
    return state.replace(data=data, obs=obs, reward=reward, done=done)

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    return jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

  # --- ricompensa ----------------------------------------------------------

  def _get_reward(self, data, action, info, metrics, done, first_contact,
                  contact) -> dict[str, jax.Array]:
    del done, first_contact, contact, metrics
    su = self.get_gravity(data, "torso")
    altezza = data.qpos[2]
    giunti = data.qpos[7:]

    errore_ori = jp.sum(jp.square(UP - su))
    in_piedi = errore_ori < self._config.reward_config.ori_tol
    h = jp.minimum(altezza, self._z_des)
    abbastanza_alto = (self._z_des - h) < self._config.reward_config.height_tol
    su_del_tutto = in_piedi * abbastanza_alto

    hn = jp.clip(h / self._z_des, 0.0, 1.0)
    testa = jp.clip(self._quota_testa(data) / self._testa_z0, 0.0, 1.0)
    vz = data.qvel[2]
    return {
        "orientation": jp.exp(-2.0 * errore_ori) * (0.3 + 0.7 * hn),
        "torso_height": hn ** 3,
        "head_height": testa ** 2,
        # salire rende, finche' non si e' arrivati (niente saltelli da in piedi)
        "up_vel": jp.clip(vz, 0.0, 1.0) * (1.0 - hn),
        "posture": su_del_tutto * jp.exp(-1.0 * jp.mean(jp.square(giunti - self._default_pose))),
        "stand_still": su_del_tutto * jp.exp(-2.0 * jp.mean(jp.square(action))),
        "action_rate": self._costo_azione(action, info),
        "torques": self._costo_coppie(data.actuator_force),
        "dof_pos_limits": self._cost_joint_pos_limits(giunti),
        "dof_acc": jp.sum(jp.square(data.qacc[6:])),
        "dof_vel": self._costo_velocita(data.qvel[6:]),
    }

  def _costo_coppie(self, coppie: jax.Array) -> jax.Array:
    return jp.sqrt(jp.sum(jp.square(coppie))) + jp.sum(jp.abs(coppie))

  def _costo_velocita(self, qvel: jax.Array) -> jax.Array:
    eccesso = jp.maximum(jp.abs(qvel) - 2.0 * jp.pi, 0.0)
    return jp.sum(jp.square(eccesso))

  def _costo_azione(self, action: jax.Array, info: dict) -> jax.Array:
    c1 = jp.sum(jp.square(action - info["last_act"]))
    c2 = jp.sum(jp.square(action - 2 * info["last_act"] + info["last_last_act"]))
    return c1 + c2


def register(attrito: float | None = None) -> None:
  """Installa il modello a collisioni complete e registra G1Getup in Playground."""
  g1_falls.install(attrito)
  from mujoco_playground._src import locomotion
  from mujoco_playground._src.locomotion.g1 import randomize as g1_randomize
  locomotion.register_environment(NOME, G1Getup, default_config)
  locomotion._randomizer[NOME] = g1_randomize.domain_randomize
