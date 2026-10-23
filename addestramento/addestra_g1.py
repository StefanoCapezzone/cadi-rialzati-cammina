"""Addestra l'Unitree G1 a camminare con PPO (MuJoCo Playground + Brax).

  python addestra_g1.py --out /content/g1_tutorial --passi 200000000 --valutazioni 21
  python addestra_g1.py --out /tmp/prova --prova --impl jax       # prova lampo (anche su CPU)
  python addestra_g1.py --out ... --set reward_config.scales.feet_air_time=0   # esperimenti

Scrive nella cartella --out:
  progress.csv          una riga per valutazione: passi, ricompensa, durata episodio, minuti
  network_config.json   come e' fatta la rete (serve per ricaricarla)
  fasi/step_*.pkl       i pesi a ogni valutazione (formato Brax: normalizzatore, attore, critico)
  fasi/step_*.npz       gli stessi pesi in numpy puro (per usarli senza JAX)
"""
import argparse, ast, csv, functools, json, os, pickle, time

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

AMBIENTE = "G1JoystickFlatTerrain"   # G1 su pavimento piatto, comandato come con un joystick

p = argparse.ArgumentParser()
p.add_argument("--out", required=True, help="cartella dei risultati")
p.add_argument("--passi", type=int, default=None, help="passi di simulazione (default DeepMind: 200M)")
p.add_argument("--valutazioni", type=int, default=21, help="quante volte misurare e salvare la rete")
p.add_argument("--impl", default="warp", choices=["warp", "jax"],
               help="motore fisico su GPU: warp (MuJoCo Warp, veloce) o jax (MJX, va anche su CPU)")
p.add_argument("--num_envs", type=int, default=None, help="robot simulati in parallelo (default 8192)")
p.add_argument("--seme", type=int, default=1)
p.add_argument("--no_dr", action="store_true", help="senza domain randomization")
p.add_argument("--set", nargs="*", default=[], metavar="CHIAVE=VALORE",
               help="modifica la configurazione dell'ambiente, es. reward_config.scales.pose=-1.0")
p.add_argument("--prova", action="store_true", help="prova lampo: pochi robot e pochi passi")
args = p.parse_args()

# ---------------------------------------------------------------------------------------------
# 1. L'ambiente: il G1 nel simulatore, con le regole del gioco (osservazioni, azioni, ricompensa)
# ---------------------------------------------------------------------------------------------
env_cfg = registry.get_default_config(AMBIENTE)
env_cfg.impl = args.impl
for kv in args.set:                       # es. reward_config.scales.feet_air_time=0
  chiave, valore = kv.split("=", 1)
  try:
    valore = ast.literal_eval(valore)
  except (ValueError, SyntaxError):
    pass
  *percorso, ultima = chiave.split(".")
  obj = env_cfg
  for parte in percorso:
    obj = obj[parte]
  obj[ultima] = valore
  print(f"[cfg] {chiave} = {valore}", flush=True)
if args.prova:
  env_cfg.naconmax = 8 * 8              # memoria per i contatti: bastano pochi robot

env = registry.load(AMBIENTE, config=env_cfg)        # ambiente per imparare
eval_env = registry.load(AMBIENTE, config=env_cfg)   # ambiente per le valutazioni
# Domain randomization: ogni robot ha attriti, masse e posa leggermente diversi
randomizer = None if args.no_dr else registry.get_domain_randomizer(AMBIENTE)

# ---------------------------------------------------------------------------------------------
# 2. PPO con gli iperparametri pubblicati da DeepMind per questo ambiente
# ---------------------------------------------------------------------------------------------
ppo_params = locomotion_params.brax_ppo_config(AMBIENTE, args.impl)
if args.passi:
  ppo_params.num_timesteps = args.passi
if args.num_envs:
  ppo_params.num_envs = args.num_envs
ppo_params.num_evals = args.valutazioni
if args.prova:
  ppo_params.update(num_timesteps=2000, num_evals=2, num_envs=8, batch_size=8,
                    num_minibatches=2, unroll_length=5, num_updates_per_batch=1)

os.makedirs(f"{args.out}/fasi", exist_ok=True)
t0 = time.monotonic()
log = open(f"{args.out}/progress.csv", "w", newline="")
logw = csv.writer(log)
logw.writerow(["step", "eval_reward", "eval_episode_length", "minutes"])


def progresso(step, metrics):
  """Chiamata da Brax dopo ogni valutazione: scrive una riga e la stampa."""
  r = float(metrics.get("eval/episode_reward", np.nan))
  L = float(metrics.get("eval/avg_episode_length", np.nan))
  m = (time.monotonic() - t0) / 60
  logw.writerow([step, r, L, round(m, 2)]); log.flush()
  print(f"[{m:6.1f} min] passi {step:>12,}  ricompensa {r:7.2f}  durata episodio {L:5.0f}/1000",
        flush=True)


def salva(nome, params):
  """Salva i pesi in due formati: .pkl (Brax/JAX) e .npz (numpy puro)."""
  params = jax.device_get(params)
  with open(f"{args.out}/fasi/{nome}.pkl", "wb") as f:
    pickle.dump(params, f)
  piatto, _ = jax.tree_util.tree_flatten_with_path(params)
  np.savez(f"{args.out}/fasi/{nome}.npz",
           **{jax.tree_util.keystr(k): np.asarray(v) for k, v in piatto})


training_params = dict(ppo_params)
net_kwargs = dict(training_params.pop("network_factory", {}) or {})
num_eval_envs = 2 if args.prova else training_params.pop("num_eval_envs", 128)
training_params.pop("num_eval_envs", None)
network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_kwargs)

with open(f"{args.out}/network_config.json", "w") as f:
  json.dump({"ambiente": AMBIENTE, "obs_size": {k: list(v) if isinstance(v, tuple) else v
                                                for k, v in dict(env.observation_size).items()},
             "action_size": env.action_size, "network_factory_kwargs": net_kwargs,
             "normalize_observations": bool(ppo_params.normalize_observations),
             "set": args.set}, f, indent=1, default=str)

print(f"[inizio] {ppo_params.num_timesteps:,} passi, {ppo_params.num_envs} robot in parallelo, "
      f"motore {args.impl}, dispositivo {jax.devices()[0].device_kind}", flush=True)

make_inference_fn, params, _ = ppo.train(
    environment=env, eval_env=eval_env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    randomization_fn=randomizer,
    network_factory=network_factory, num_eval_envs=num_eval_envs,
    seed=args.seme, progress_fn=progresso,
    policy_params_fn=lambda step, mk, prm: salva(f"step_{step:012d}", prm),
    **training_params)
salva("step_final", params)
print(f"Fatto in {(time.monotonic() - t0) / 60:.1f} minuti. Pesi in {args.out}/fasi", flush=True)
