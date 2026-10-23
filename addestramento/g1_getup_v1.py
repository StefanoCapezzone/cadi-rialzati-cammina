"""G1Getup: il G1 parte a terra e deve rimettersi in piedi.

E' una sottoclasse dell'ambiente di camminata di Playground (g1.joystick.Joystick),
non un ambiente nuovo: cosi' la rete ha le stesse identiche entrate e uscite di
quella che cammina, e tutti gli script di rendering funzionano senza modifiche
(in prospettiva si possono anche incatenare: cade -> si rialza -> riparte).

Cambiano tre cose rispetto alla camminata:
  1. reset(): con probabilita' 0.8 il robot parte a terra (lasciato cadere da 45 cm
     con orientamento casuale, poi assestato mezzo secondo), altrimenti in piedi.
     Il misto e' lo stesso trucco che ha fatto funzionare lo stickman.
  2. ricompensa: quella di Go1Getup (orientamento, altezza del bacino, postura,
     stai fermo una volta su) piu' i soliti termini di regolarizzazione.
  3. nessuna terminazione anticipata: cadere non e' un fallimento, e' il punto
     di partenza. Si ferma solo se la simulazione diverge.

ATTENZIONE al segno: nel G1 get_gravity() legge il sensore "upvector"
(framezaxis), che vale [0,0,+1] quando il robot e' in piedi. Nel Go1 la stessa
funzione proietta la gravita' nel sistema dell'IMU e vale [0,0,-1]. Copiando la
ricompensa del Go1 senza cambiare il segno il robot impara a stare a testa in giu'.

Uso:
  import g1_getup; g1_getup.register()      # installa anche il modello a collisioni complete
  env = registry.load("G1Getup")
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

import g1_falls

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion import g1 as _g1_pkg  # noqa: F401
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick

NOME = "G1Getup"
UP = jp.array([0.0, 0.0, 1.0])   # il sensore upvector del G1: [0,0,1] = in piedi
ALTEZZA_IN_PIEDI = 0.755         # bacino nella posa "knees_bent" del modello


def default_config() -> config_dict.ConfigDict:
  cfg = g1_joystick.default_config()

  # Episodi corti: sei secondi bastano per rialzarsi.
  cfg.episode_length = 300

  # Niente spinte e niente comandi di velocita': l'unico compito e' alzarsi.
  # ATTENZIONE: sample_command() ignora command_config e pesca dagli intervalli
  # qui sotto, quindi sono questi che vanno azzerati.
  cfg.push_config.enable = False
  cfg.lin_vel_x = [0.0, 0.0]
  cfg.lin_vel_y = [0.0, 0.0]
  cfg.ang_vel_yaw = [0.0, 0.0]

  # Il corpo tocca terra dappertutto: servono piu' contatti di quelli della camminata.
  cfg.naconmax = 24 * 8192
  cfg.njmax = 29 * 2 + 24 * 4

  # Da dove si parte.
  cfg.drop_from_height_prob = 0.8
  cfg.drop_height = 0.45
  cfg.settle_time = 0.6
  cfg.joint_spread = 0.5      # quanto si allontanano i giunti dalla posa di riposo
  cfg.height_target = 0.68    # bacino: sotto i 0.755 in piedi, cosi' e' raggiungibile

  # Ricompensa: quella di Go1Getup, adattata.
  cfg.reward_config = config_dict.create(
      scales=config_dict.create(
          orientation=1.0,
          torso_height=1.0,
          posture=1.0,
          stand_still=1.0,
          action_rate=-0.001,
          torques=-1e-5,
          dof_pos_limits=-0.1,
          dof_acc=-2.5e-7,
          dof_vel=-0.05,
      ),
      ori_tol=0.05,      # quanto "in piedi" conta come in piedi
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
    self._z_des = self._config.height_target
    self._mezzo_intervallo = 0.5 * (self._uppers - self._lowers)

  # --- da dove si parte ----------------------------------------------------

  def _posa_a_terra(self, rng: jax.Array) -> jax.Array:
    """Robot lasciato cadere da 45 cm con orientamento e giunti a caso."""
    rng, k_or, k_q = jax.random.split(rng, 3)
    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[2].set(self._config.drop_height)
    quat = jax.random.normal(k_or, (4,))
    quat = quat / (jp.linalg.norm(quat) + 1e-6)
    qpos = qpos.at[3:7].set(quat)
    # giunti attorno alla posa di riposo, non su tutto l'intervallo: pose meno assurde
    delta = jax.random.uniform(k_q, (29,), minval=-1.0, maxval=1.0)
    giunti = self._default_pose + delta * self._config.joint_spread * self._mezzo_intervallo
    qpos = qpos.at[7:].set(jp.clip(giunti, self._soft_lowers, self._soft_uppers))
    return qpos

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Si parte dallo stato costruito dal genitore: cosi' info e metriche hanno
    # tutte le chiavi che step() si aspetta (fase del passo, spinte, contatti...).
    state = super().reset(rng)
    rng, k_scelta, k_posa, k_vel = jax.random.split(state.info["rng"], 4)

    qpos = jp.where(
        jax.random.bernoulli(k_scelta, self._config.drop_from_height_prob),
        self._posa_a_terra(k_posa),
        self._init_q,
    )
    qvel = jp.zeros(self.mjx_model.nv)
    qvel = qvel.at[0:6].set(jax.random.uniform(k_vel, (6,), minval=-0.5, maxval=0.5))

    data = mjx_env.make_data(
        self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax, njmax=self._config.njmax,
    )
    # lo si lascia assestare a motori fermi, poi l'episodio comincia davvero
    data = mjx_env.step(self.mjx_model, data, qpos[7:], self._settle_steps)
    data = data.replace(time=0.0)

    info = dict(state.info)
    info["rng"] = rng
    info["command"] = jp.zeros(3)

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sid]] > 0
        for sid in self._feet_floor_found_sensor
    ])
    obs = self._get_obs(data, info, contact)
    return state.replace(data=data, obs=obs, info=info)

  # --- cadere non e' un fallimento -----------------------------------------

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    return jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

  # --- ricompensa ----------------------------------------------------------

  def _get_reward(self, data, action, info, metrics, done, first_contact,
                  contact) -> dict[str, jax.Array]:
    del done, first_contact, contact, metrics

    su = self.get_gravity(data, "torso")      # [0,0,1] quando e' in piedi
    altezza = data.qpos[2]                    # altezza del bacino
    giunti = data.qpos[7:]

    errore_ori = jp.sum(jp.square(UP - su))
    in_piedi = errore_ori < self._config.reward_config.ori_tol
    h = jp.minimum(altezza, self._z_des)
    abbastanza_alto = (self._z_des - h) < self._config.reward_config.height_tol
    su_del_tutto = in_piedi * abbastanza_alto

    return {
        "orientation": jp.exp(-2.0 * errore_ori),
        "torso_height": jp.exp(h) - 1.0,
        "posture": in_piedi * jp.exp(-1.0 * jp.mean(jp.square(giunti - self._default_pose))),
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
    """Scatti: quanto l'azione cambia da un passo all'altro (prima e seconda differenza)."""
    c1 = jp.sum(jp.square(action - info["last_act"]))
    c2 = jp.sum(jp.square(action - 2 * info["last_act"] + info["last_last_act"]))
    return c1 + c2


def register(attrito: float | None = None) -> None:
  """Installa il modello a collisioni complete e registra G1Getup in Playground."""
  g1_falls.install(attrito)
  from mujoco_playground._src import locomotion
  from mujoco_playground._src.locomotion.g1 import randomize as g1_randomize
  locomotion.register_environment(NOME, G1Getup, default_config)
  # stesso robot della camminata: va bene lo stesso randomizzatore
  locomotion._randomizer[NOME] = g1_randomize.domain_randomize
