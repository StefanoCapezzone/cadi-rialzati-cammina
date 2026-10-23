"""Addestra un ambiente di locomozione di MuJoCo Playground (es. G1JoystickFlatTerrain)
con i parametri PPO ufficiali di DeepMind e salva le "fasi" dell'apprendimento.

Uso (Colab, GPU):  python train_loco.py --env G1JoystickFlatTerrain --out /content/drive/MyDrive/linuxday/g1_walk
Prova su CPU:      python train_loco.py --env G1JoystickFlatTerrain --smoke --impl jax --out /tmp/g1smoke
Ambiente su misura: python train_loco.py --env G1Getup --register g1_getup \
                      --ppo_env G1JoystickFlatTerrain --out .../g1_getup
"""
import argparse, csv, functools, os, pickle, time

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

p = argparse.ArgumentParser()
p.add_argument("--env", default="G1JoystickFlatTerrain")
p.add_argument("--out", required=True)
p.add_argument("--impl", default="warp", choices=["warp", "jax"])
p.add_argument("--timesteps", type=int, default=None, help="default: quello di DeepMind")
p.add_argument("--evals", type=int, default=41, help="numero di fasi salvate")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--no_dr", action="store_true", help="senza domain randomization")
p.add_argument("--smoke", action="store_true", help="prova minima su CPU")
p.add_argument("--register", default=None,
               help="modulo con un ambiente su misura da registrare prima di caricarlo "
                    "(es. g1_getup): ne viene chiamata la funzione register()")
p.add_argument("--ppo_env", default=None,
               help="da quale ambiente prendere i parametri PPO, se il nostro non e' "
                    "nel listino di DeepMind (es. G1JoystickFlatTerrain)")
p.add_argument("--set", nargs="*", default=[], metavar="CHIAVE=VALORE",
               help="modifica la configurazione dell'ambiente (es. assist=0.6)")
p.add_argument("--ppo_set", nargs="*", default=[], metavar="CHIAVE=VALORE",
               help="modifica i parametri PPO (es. discounting=0.99)")
p.add_argument("--restore", default=None,
               help="riparte dai pesi di una fase salvata (.pkl = normalizzatore, policy, value)")
p.add_argument("--no_restore_value", action="store_true",
               help="con --restore: il critico riparte da zero (ricompensa diversa)")
args = p.parse_args()


def coppie(lista):
  import ast
  out = {}
  for kv in lista:
    k, v = kv.split("=", 1)
    try:
      out[k] = ast.literal_eval(v)
    except (ValueError, SyntaxError):
      out[k] = v
  return out

if args.register:
  import importlib
  importlib.import_module(args.register).register()

env_cfg = registry.get_default_config(args.env)
env_cfg.impl = args.impl
ppo_params = locomotion_params.brax_ppo_config(args.ppo_env or args.env, args.impl)
# la lunghezza dell'episodio e' quella del NOSTRO ambiente, non di quello dei parametri
ppo_params.episode_length = env_cfg.episode_length
if args.timesteps:
  ppo_params.num_timesteps = args.timesteps
ppo_params.num_evals = args.evals
for k, v in coppie(args.set).items():
  env_cfg[k] = v
for k, v in coppie(args.ppo_set).items():
  ppo_params[k] = v
print(f"[cfg] ambiente: {coppie(args.set)}  ppo: {coppie(args.ppo_set)}  restore: {args.restore}", flush=True)
if args.smoke:
  env_cfg.naconmax = 8 * 8
  ppo_params.update(num_timesteps=2000, num_evals=2, num_envs=8, batch_size=8,
                    num_minibatches=2, unroll_length=5, num_updates_per_batch=1)
env = registry.load(args.env, config=env_cfg)
eval_env = registry.load(args.env, config=env_cfg)
randomizer = None if args.no_dr else registry.get_domain_randomizer(args.env)

os.makedirs(f"{args.out}/stages", exist_ok=True)
t0 = time.monotonic()
log = open(f"{args.out}/progress.csv", "w", newline="")
logw = csv.writer(log)
logw.writerow(["step", "eval_reward", "eval_episode_length", "minutes"])


def progress(step, metrics):
  r = float(metrics.get("eval/episode_reward", np.nan))
  L = float(metrics.get("eval/avg_episode_length", np.nan))
  m = (time.monotonic() - t0) / 60
  logw.writerow([step, r, L, round(m, 2)]); log.flush()
  print(f"[{m:6.1f} min] step {step:>12,}  ricompensa {r:8.2f}  durata episodio {L:6.0f}", flush=True)


def to_numpy_tree(tree):
  flat, _ = jax.tree_util.tree_flatten_with_path(tree)
  return {jax.tree_util.keystr(k): np.asarray(v) for k, v in flat}


def save(name, params):
  with open(f"{args.out}/stages/{name}.pkl", "wb") as f:
    pickle.dump(jax.device_get(params), f)
  np.savez(f"{args.out}/stages/{name}.npz", **to_numpy_tree(params))


training_params = dict(ppo_params)
net_kwargs = dict(training_params.pop("network_factory", {}) or {})
num_eval_envs = training_params.pop("num_eval_envs", 128)
if args.smoke:
  num_eval_envs = 2
network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_kwargs)

with open(f"{args.out}/network_config.pkl", "wb") as f:
  pickle.dump({"env": args.env, "env_set": coppie(args.set), "obs_size": env.observation_size,
               "action_size": env.action_size, "network_factory_kwargs": net_kwargs,
               "normalize_observations": ppo_params.normalize_observations}, f)

extra = {}
if args.restore:
  with open(args.restore, "rb") as f:
    extra["restore_params"] = pickle.load(f)
  extra["restore_value_fn"] = not args.no_restore_value

make_inference_fn, params, _ = ppo.train(
    environment=env, eval_env=eval_env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    randomization_fn=randomizer,
    network_factory=network_factory, num_eval_envs=num_eval_envs,
    seed=args.seed, progress_fn=progress,
    policy_params_fn=lambda step, mk, prm: save(f"step_{step:012d}", prm),
    **training_params, **extra)
save("step_final", params)
print(f"Fatto in {(time.monotonic()-t0)/60:.1f} minuti. Fasi in {args.out}/stages", flush=True)
