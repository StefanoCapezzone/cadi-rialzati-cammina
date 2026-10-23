"""Addestra HumanoidGetup con Brax PPO e salva le "fasi" dell'apprendimento.

Uso (Colab con GPU):   python train_getup.py --out /content/drive/MyDrive/linuxday/run1
Prova veloce su CPU:   python train_getup.py --smoke --impl jax --out /tmp/smoke

Output nella cartella --out:
  stages/step_<N>.pkl   parametri della rete a ogni valutazione (una "fase")
  stages/step_<N>.npz   gli stessi parametri in numpy (per il Mac, senza JAX)
  progress.csv          passi, ricompensa media, altezza della testa, minuti
  (i video si fanno dopo, con render_stages.py)
"""
import argparse, csv, functools, os, pickle, time

import jax
import jax.numpy as jp
import numpy as np

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import dm_control_suite_params

import humanoid_getup

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--impl", default="warp", choices=["warp", "jax"])
p.add_argument("--timesteps", type=int, default=60_000_000)
p.add_argument("--evals", type=int, default=25, help="numero di fasi salvate")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--resets_per_eval", type=int, default=1,
               help="1 = una fase ogni ~1M passi (default DeepMind: 10 = ogni ~10M)")
p.add_argument("--smoke", action="store_true", help="prova minima su CPU")
args = p.parse_args()

humanoid_getup.register()
ENV = "HumanoidGetup"
env_cfg = registry.get_default_config(ENV)
env_cfg.impl = args.impl
if args.smoke:
  env_cfg.num_fallen_poses = 16
env = registry.load(ENV, config=env_cfg)

# Iperparametri tarati da DeepMind per HumanoidStand, adattati.
ppo_params = dm_control_suite_params.brax_ppo_config("HumanoidStand", args.impl)
ppo_params.num_timesteps = args.timesteps
ppo_params.num_evals = args.evals
ppo_params.episode_length = env_cfg.episode_length
ppo_params.num_resets_per_eval = args.resets_per_eval
if args.smoke:
  ppo_params.update(num_timesteps=4000, num_evals=2, num_envs=8, batch_size=8,
                    num_minibatches=2, unroll_length=10, num_updates_per_batch=1)
  ppo_params.num_resets_per_eval = 0

os.makedirs(f"{args.out}/stages", exist_ok=True)
t0 = time.monotonic()
log = open(f"{args.out}/progress.csv", "w", newline="")
logw = csv.writer(log)
logw.writerow(["step", "eval_reward", "eval_head_height", "minutes"])


def progress(step, metrics):
  r = float(metrics.get("eval/episode_reward", np.nan))
  h = float(metrics.get("eval/episode_head_height", np.nan))
  m = (time.monotonic() - t0) / 60
  ep_len = ppo_params.episode_length
  logw.writerow([step, r, h / ep_len if h == h else h, round(m, 2)]); log.flush()
  print(f"[{m:6.1f} min] step {step:>11,}  ricompensa {r:8.1f}", flush=True)


def to_numpy_tree(tree):
  flat, _ = jax.tree_util.tree_flatten_with_path(tree)
  return {jax.tree_util.keystr(k): np.asarray(v) for k, v in flat}


def policy_params_fn(step, make_policy, params):
  with open(f"{args.out}/stages/step_{step:011d}.pkl", "wb") as f:
    pickle.dump(jax.device_get(params), f)
  np.savez(f"{args.out}/stages/step_{step:011d}.npz", **to_numpy_tree(params))


training_params = dict(ppo_params)
net_kwargs = training_params.pop("network_factory", {}) or {}
num_eval_envs = training_params.pop("num_eval_envs", 128)
if args.smoke:
  num_eval_envs = 2
network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_kwargs)

make_inference_fn, params, _ = ppo.train(
    environment=env,
    eval_env=registry.load(ENV, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
    network_factory=network_factory,
    num_eval_envs=num_eval_envs,
    seed=args.seed,
    progress_fn=progress,
    policy_params_fn=policy_params_fn,
    **training_params,
)
with open(f"{args.out}/stages/step_final.pkl", "wb") as f:
  pickle.dump(jax.device_get(params), f)
np.savez(f"{args.out}/stages/step_final.npz", **to_numpy_tree(params))
with open(f"{args.out}/network_config.pkl", "wb") as f:
  pickle.dump({"obs_size": env.observation_size, "action_size": env.action_size,
               "network_factory_kwargs": dict(net_kwargs),
               "normalize_observations": ppo_params.normalize_observations}, f)
print(f"Fatto in {(time.monotonic()-t0)/60:.1f} minuti. Fasi in {args.out}/stages")
