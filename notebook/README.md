# Il tutorial

[![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StefanoCapezzone/cadi-rialzati-cammina/blob/main/notebook/cadi_rialzati_cammina_tutorial.ipynb)

`cadi_rialzati_cammina_tutorial.ipynb` addestra da zero la rete che fa camminare il G1, spiegando ogni passaggio. Si apre direttamente in Google Colab dal pulsante qui sopra: il notebook viene copiato nel tuo account e gira sulla GPU che scegli.

Sezioni: GPU, tempi e costi · installazione · il G1 nel simulatore · cosa vede e cosa comanda la rete · la ricompensa · un robot senza cervello (video) · PPO in breve, con le formule · gli iperparametri · **l'addestramento** · le curve di apprendimento · il robot addestrato cammina (video) · com'è cambiato durante l'addestramento (video affiancati) · la rete in numpy · esperimenti da provare · bibliografia.

La modalità si sceglie da sola in base alla GPU: `prova` (anche su CPU, solo verifica), `veloce` (T4 gratuita, 50 milioni di passi, ~35 minuti) o `completo` (A100/L4/H100, 200 milioni di passi come DeepMind, ~40 minuti su A100). Tempi misurati il 24 ottobre 2026.

## Versione già eseguita
`cadi_rialzati_cammina_tutorial_eseguito.ipynb` è lo stesso notebook eseguito su una A100 (203 milioni di passi in 40 minuti, ricompensa finale 14,6), con tutti gli output: grafici e video. Si apre in [Colab](https://colab.research.google.com/github/StefanoCapezzone/cadi-rialzati-cammina/blob/main/notebook/cadi_rialzati_cammina_tutorial_eseguito.ipynb) o su [nbviewer](https://nbviewer.org/github/StefanoCapezzone/cadi-rialzati-cammina/blob/main/notebook/cadi_rialzati_cammina_tutorial_eseguito.ipynb); l'anteprima di GitHub mostra i grafici ma non i video.
