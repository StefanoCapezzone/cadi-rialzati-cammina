# Lo stickman che si rialza

Il "riscaldamento" del progetto, prima del G1: l'umanoide stilizzato della DeepMind Control Suite (in MuJoCo Playground) parte sdraiato a terra e deve rialzarsi.

| script | cosa fa |
|---|---|
| `humanoid_getup.py` | l'ambiente `HumanoidGetup`: variante di `dm_control_suite/humanoid.py` di MuJoCo Playground (Apache-2.0). Parte da pose "cadute" generate con MuJoCo e premia lo stare in piedi, con un piccolo incentivo ad alzare la testa |
| `train_getup.py` | addestramento con PPO di Brax (gira anche su una T4) |
| `render_stages.py` | video delle fasi salvate |

Tre run: la prima finiva in ginocchio o seduto, la seconda (ricompensa "cubica") accovacciato, la terza (con il 30% di partenze in piedi) **si alza in circa 1,5 secondi** e resta in piedi, con pose buffe.
