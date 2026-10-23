# Gli script dell'addestramento

Gli script che abbiamo usato davvero, su Google Colab con GPU A100 (settembre-ottobre 2026). Stanno tutti nella stessa cartella perché si importano a vicenda (`render_walks.py` usa `render_loco.py`, `g1_getup.py` usa `g1_falls.py`, …).

**Se vuoi solo ripetere l'addestramento, usa il [notebook](../notebook/)**: è autosufficiente e spiega tutto. Questi script servono a chi vuole andare più a fondo.

## Installazione (Colab)
```bash
pip install -U "playground==0.2.0" "jax[cuda12]==0.9.2" "jaxlib==0.9.2" "flax==0.12.6" mediapy
pip uninstall -y jax-cuda13-plugin jax-cuda13-pjrt     # Colab da ottobre 2026: evita il conflitto tra plugin CUDA
```

## La camminata
| script | cosa fa |
|---|---|
| `addestra_g1.py` | versione didattica e commentata dell'addestramento (è quella del notebook). `--passi`, `--valutazioni`, `--set reward_config.scales.X=…`, `--no_dr`, `--prova` |
| `train_loco.py` | lo script originale: addestra un ambiente di locomozione con i parametri PPO di DeepMind e salva una "fase" (pesi `.pkl` + `.npz`) a ogni valutazione. Supporta ambienti su misura (`--register`), modifiche alla configurazione (`--set`, `--ppo_set`) e la ripartenza da pesi salvati (`--restore`) |
| `render_loco.py` | video di tutte le fasi con lo stesso percorso di comandi (fermo, avanti, curva, di lato) |
| `render_walks.py` | tre camminate diverse della rete finale (avanti e curva / di lato e indietro / veloce): la clip 4 |

La nostra run della camminata: `python train_loco.py --env G1JoystickFlatTerrain --evals 41 --out …/g1_walk` → 203 milioni di passi in 124 minuti.

## Le cadute (per i video)
Il modello del G1 usato in addestramento tocca il pavimento solo con i piedi: quando cade, ci sprofonda dentro. Per i video delle cadute serve un modello con le collisioni su tutto il corpo.
| script | cosa fa |
|---|---|
| `g1_falls.py` | G1 con collisioni complete (bacino, busto, testa, braccia); `install(attrito=0.2)` = pavimento ghiacciato |
| `render_falls.py` | video delle cadute: quando cade, i motori si spengono e il robot resta a terra qualche secondo, poi riparte |
| `falls_scan.py` | cerca le cadute "buone" simulando molte combinazioni (fase, velocità, attrito) **senza** rendere i video |
| `episodi.py` | ricava gli istanti delle cadute dai video già fatti; il catalogo risultante è `cadute_istanti.md` |
| `clip_demo.py` | monta le sei clip del talk (didascalie, contatore dei passi) a partire dal catalogo |

## Rialzati: insegnare al G1 ad alzarsi da terra
Un ambiente su misura, `G1Getup`, sottoclasse della camminata (stessi 103 ingressi, 29 uscite). Tre versioni, tre lezioni:

| versione | ricompensa / modifica | risultato |
|---|---|---|
| `g1_getup_v1.py` | premio per alzare il busto | dopo 152M passi si **sdraia sui gomiti** e resta lì |
| `g1_getup_v2.py` | premio per l'altezza, partenze più varie | dopo 302M passi si **siede composto** (bacino a 0,43 m; in piedi è 0,755) |
| `g1_getup.py` (v3) | azione su **tutta l'escursione** dei giunti (prima ±0,5 rad: le ginocchia non potevano piegarsi abbastanza), una **"mano" che lo tira su** (forza verticale sul bacino, tolta a tappe: 0,6 → 0,3 → 0,1 → 0), sconto 0,99, premio per l'altezza della testa e la velocità verso l'alto | dopo 450M passi in 4 fasi **si alza in circa un secondo e resta in piedi**, senza aiuto |

- `run_v3.sh` lancia le quattro fasi della v3 una dopo l'altra (ciascuna riparte dai pesi della precedente) e fa i video a fine fase; se il runtime si interrompe, riprende da dove era rimasto.
- `render_getup.py` fa i video dei tentativi (`--partenza terra|accovacciato`) e scrive le altezze del bacino in `videos_getup/altezze.txt`: è lì che si giudica se si è alzato davvero.

Esempio di una fase a mano:
```bash
python train_loco.py --env G1Getup --register g1_getup --ppo_env G1JoystickFlatTerrain \
    --timesteps 100000000 --evals 8 --out …/g1_getup_v3/f2 --set assist=0.3 \
    --ppo_set discounting=0.99 entropy_cost=0.01 --restore …/f1/stages/step_final.pkl
```

## Cose che abbiamo imparato a nostre spese
- **Guarda i video, non solo la ricompensa**: nel "rialzati" la ricompensa saliva anche quando il robot aveva solo trovato una posa comoda.
- Il rendering è la parte lenta (~0,5 s per fotogramma su A100); la simulazione no. I video conviene farli in un processo separato dall'addestramento, per non esaurire la memoria della GPU.
- In MuJoCo Warp, per unire due `Data` si usa `d1.where(cond, d2)`, non `jax.tree.map(jp.where, …)`.
- Una posa di partenza impostata nel reset può crollare durante l'assestamento: controlla l'altezza **reale** di partenza prima di fidarti di un curriculum.
- I messaggi *iterations limit reached* di MuJoCo Warp durante l'addestramento sono innocui.

I percorsi negli esempi (`/content/drive/MyDrive/linuxday/…`) sono quelli del nostro Google Drive: sostituiscili con i tuoi.
