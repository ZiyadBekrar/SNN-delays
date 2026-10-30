# Étude des délais axonaux, synaptiques et hybrides pour la reconnaissance d’activités humaines

Cette section présente les expériences HAR des diapositives 5 à 18 de `SNN-etude-de-cas.pdf`. Les résultats numériques sont retranscrits depuis les diapositives ; le protocole est décrit à partir des scripts et de la configuration présents dans le dépôt. Cette rédaction ne constitue pas une nouvelle exécution des expériences. Les points nécessitant une vérification des runs d’origine sont regroupés en fin de section.

## Objectif et formulations comparées

Nous étudions l’influence de la paramétrisation des délais sur la reconnaissance d’activités humaines, en comparant des réseaux de neurones impulsionnels sans récurrence (*feedforward*) et récurrents. L’objectif est de déterminer si une hétérogénéité temporelle propre à chaque connexion améliore la classification sans augmenter le nombre de paramètres appris par rapport à une formulation axonale.

Pour une connexion allant du neurone présynaptique \(j\) au neurone postsynaptique \(i\), les trois formulations s’écrivent :

\[
d^{\mathrm{ax}}_{ij}=a_j,\qquad
d^{\mathrm{syn}}_{ij}=s_{ij},\qquad
d^{\mathrm{hyb}}_{ij}=a_j+\delta_{ij}.
\]

Dans le modèle axonal, un délai appris est partagé par toutes les connexions sortantes d’un neurone. Le modèle synaptique apprend un délai indépendant par connexion. Le modèle hybride conserve un délai axonal appris, auquel s’ajoute un décalage synaptique entier, tiré une seule fois selon une loi uniforme discrète sur \(\{0,1,2,3,4\}\), puis maintenu fixe pendant l’entraînement. Il introduit ainsi une diversité temporelle entre connexions sans ajouter de délais entraînables. Les décalages fixes nécessitent néanmoins du stockage : l’égalité du nombre de paramètres appris ne signifie pas une égalité de consommation mémoire ou de temps de calcul.

Dans l’implémentation récurrente, ces quantités représentent le délai additionnel : un spike émis à l’instant \(t\) est transmis avec une latence de \(1+d_{ij}\) pas de temps.

## Données et protocole d’entraînement

La tâche utilise WISDM, avec 18 classes d’activités et les trois axes du gyroscope de la montre. Le prétraitement standardise chaque axe par fichier, puis construit des fenêtres de 200 pas de temps avec un recouvrement de 100 pas. Les fenêtres dont les étiquettes initiale et finale diffèrent sont écartées. Chaque entrée du réseau est donc une séquence de dimension \(200\times3\).

Le chargeur réalise une séparation aléatoire des fenêtres en 80 % pour l’apprentissage et 20 % pour l’évaluation, avec `random_state=42`, puis conserve ces ensembles dans des fichiers NumPy. Lorsque ces fichiers existent, ils sont directement rechargés. Ce protocole ne réalise pas une séparation par sujet. De plus, les fenêtres se recouvrant avant la séparation, des observations temporellement proches peuvent se retrouver dans les deux ensembles.

Les réseaux utilisent trois couches cachées de 128, 176 et 176 neurones, puis une sortie à 18 classes. Dans les architectures récurrentes, seules les deux premières couches cachées portent des connexions récurrentes ; la dernière est une couche LIF sans récurrence. La configuration emploie des neurones LIF avec un seuil de 0,5, une remise à zéro après émission et un facteur de rétention membranaire de 0,1. L’apprentissage utilise la rétropropagation temporelle avec un gradient de substitution triangulaire.

Les paramètres communs de la configuration HAR sont les suivants :

| Paramètre | Valeur |
|---|---:|
| Époques | 100 |
| Taille de batch | 256 |
| Optimiseur | AdamW |
| Taux d’apprentissage des poids | \(1{,}5\times10^{-3}\) |
| Taux d’apprentissage des positions/délais | \(5\times10^{-2}\) |
| Décroissance des taux | facteur 0,8 toutes les 10 époques |
| Weight decay des poids et des délais | 0 |
| Dropout récurrent | 0,2 |
| Dropout feedforward | 0 |

La fonction de coût est une entropie croisée appliquée à la moyenne temporelle des sorties. La classe prédite est l’indice de la plus grande composante de cette moyenne. Les états neuronaux sont réinitialisés à chaque batch. Pour les délais récurrents, la configuration active une projection sur les entiers avant chaque évaluation. Les délais feedforward reposent sur des noyaux DCLS gaussiens dont la largeur décroît jusqu’à 0,23 pendant la première moitié de l’entraînement. Cette largeur de noyau est distincte du paramètre d’initialisation étudié plus loin.

Les scripts de comparaison prévoient trois graines d’apprentissage, 0, 1 et 2. La génération des décalages hybrides utilise une graine indépendante, fixée par défaut à 123. Dans l’ablation à délais fixes, le lanceur permet au contraire de faire varier les graines des décalages en conservant la graine d’apprentissage : ces deux sources de variabilité doivent être distinguées lors de l’agrégation.

À chaque époque, le programme enregistre les métriques et sauvegarde le meilleur checkpoint selon l’accuracy d’évaluation. Il recharge ce checkpoint pour produire `final_test.json`. Dans le chargeur HAR actuel, les ensembles appelés « validation » et « test » sont identiques. Les scores présentés correspondent donc à une évaluation utilisée également pour la sélection du modèle ; ils ne doivent pas être décrits comme les résultats d’un test indépendant tenu à l’écart de cette sélection.

## Comparaison globale des architectures — diapositives 5 à 8

Le tableau suivant reprend les résultats de la diapositive 6. Les dispersions sont conservées telles qu’elles sont affichées dans le support.

| Architecture | Délais | Paramètres rapportés | Accuracy (%) |
|---|---|---:|---:|
| Feedforward | Axonaux | 58 037 | 78,986 ± 0,680 |
| Feedforward | Hybrides | 58 037 | 80,291 ± 0,758 |
| Feedforward | Synaptiques | 171 666 | 81,632 ± 0,340 |
| Récurrente | Axonaux | 105 522 | 82,896 ± 0,133 |
| Récurrente | Hybrides | 105 522 | 82,855 ± 0,537 |
| Récurrente | Synaptiques | 152 578 | 83,709 ± 0,188 |

En feedforward, la formulation hybride améliore l’accuracy moyenne de 1,305 point par rapport à la formulation axonale, à nombre de paramètres rapporté identique. Les courbes de la diapositive 7 montrent que deux exécutions hybrides dépassent les courbes axonales, tandis que la troisième reste proche de la meilleure exécution axonale. Le modèle synaptique obtient le meilleur score moyen, au prix d’un nombre de paramètres environ 2,96 fois plus élevé que celui des variantes axonale et hybride.

Dans les réseaux récurrents, l’écart entre hybride et axonal est de −0,041 point. Les courbes se recouvrent et ne montrent pas d’avantage net pour l’hybride. Le modèle synaptique atteint la meilleure moyenne, avec environ 45 % de paramètres supplémentaires par rapport au modèle axonal. Les résultats suggèrent donc que l’intérêt des décalages fixes dépend de l’architecture. L’hypothèse selon laquelle la récurrence fournit déjà une partie de la diversité temporelle utile est compatible avec ces observations, mais n’est pas directement démontrée par cette comparaison.

## Vérification des cas limites du modèle hybride — diapositives 9 à 13

Deux ablations permettent de vérifier que la formulation hybride retrouve les comportements attendus lorsque l’une de ses composantes est supprimée.

### Partie axonale fixée à zéro

En imposant \(a_j=0\) pendant tout l’entraînement, on obtient \(d^{\mathrm{hyb}}_{ij}=\delta_{ij}\). Le réseau possède alors uniquement des délais synaptiques fixes ; les poids restent appris. La référence utilise également des délais synaptiques fixes tirés uniformément parmi les entiers de 0 à 4. Les distributions sont identiques, sans que les matrices tirées soient nécessairement les mêmes.

La diapositive 10 rapporte 80,470 ± 0,893 % pour l’hybride à partie axonale nulle, contre 80,669 ± 0,669 % pour le synaptique fixe. L’écart de 0,199 point et le recouvrement des courbes de la diapositive 11 sont cohérents avec cette équivalence de formulation. Ils ne constituent pas un test d’identité numérique entre implémentations. Les variantes à délais appris atteignent respectivement 82,855 ± 0,537 % et 83,7095 ± 0,1880 % dans ce tableau, ce qui indique une amélioration lorsque les délais peuvent être adaptés.

### Décalages synaptiques fixés à zéro

Lorsque \(\delta_{ij}=0\), le modèle hybride se réduit à \(d^{\mathrm{hyb}}_{ij}=a_j\). L’implémentation conserve l’architecture hybride, mais chaque connexion issue d’un même neurone partage le même délai appris.

La diapositive 12 rapporte 82,926 ± 0,771 % pour l’hybride sans décalages, contre 82,896 ± 0,162 % pour l’axonal appris. L’écart moyen de 0,030 point et le recouvrement des courbes de la diapositive 13 sont compatibles avec le comportement attendu. Cette dernière diapositive porte un titre mentionnant « Synaptic », mais sa légende et l’ablation précédente indiquent bien une comparaison hybride–axonal.

## Influence de l’initialisation axonale — diapositives 14 à 18

L’analyse des délais finaux motive une dernière expérience. La diapositive 15 agrège les délais de trois graines et de deux couches récurrentes :

| Délais | Moyenne | Écart-type | Médiane | Min–max | Délais nuls (%) |
|---|---:|---:|---:|---:|---:|
| Hybrides | 6,28 | 4,52 | 5 | 0–27 | 3,40 |
| Synaptiques | 3,48 | 3,47 | 3 | 0–30 | 23,93 |
| Axonaux | 5,06 | 4,52 | 4 | 0–24 | 9,32 |

Ces valeurs sont exprimées en pas de temps. Pour l’hybride, elles comprennent la composante axonale et le décalage synaptique. Leur amplitude supérieure ne peut donc pas être attribuée uniquement à l’apprentissage : les décalages fixes augmentent déjà les délais effectifs à l’initialisation.

Pour étudier cette dépendance, la partie axonale est initialisée selon une loi demi-normale :

\[
a_j^{(0)}=|Z_j|,\qquad Z_j\sim\mathcal{N}(0,\sigma_{\mathrm{init}}^2),
\qquad \sigma_{\mathrm{init}}\in\{0,1,2,3\}.
\]

Ici, \(\sigma_{\mathrm{init}}\) est l’écart-type de la normale avant valeur absolue, et non l’écart-type de la demi-normale résultante. Il correspond à `delay_std_init`, et non à la largeur du noyau temporel. Les décalages synaptiques fixes restent présents. Pour \(\sigma_{\mathrm{init}}=0\), la partie axonale démarre à zéro ; cela ne signifie pas qu’elle est gelée, contrairement à l’ablation précédente.

Les diapositives 17 et 18 rapportent :

| \(\sigma_{\mathrm{init}}\) | Délai effectif moyen | Écart-type des délais | Min–max | Accuracy sur 3 graines (%) |
|---:|---:|---:|---:|---:|
| 0 | 1,998 | 1,414 | 0–4 | 80,153 ± 0,723 |
| 1 | 5,606 | 4,308 | 0–26 | 82,568 ± 0,288 |
| 2 | 5,941 | 4,378 | 0–25 | 82,645 ± 0,366 |
| 3 | 6,276 | 4,516 | 0–27 | 82,558 ± 0,614 |

Entre 1 et 3, une initialisation plus large produit des délais finaux moyens plus grands, tandis que les accuracies moyennes restent très proches : leur étendue n’est que de 0,087 point. Dans cette plage, l’amplitude des délais est donc sensible à l’initialisation sans amélioration mesurable évidente de la classification sur les trois graines présentées.

Le cas nul se distingue : son accuracy est inférieure de 2,415 à 2,492 points aux trois initialisations non nulles. Les statistiques finales de délai sont proches de celles des seuls décalages uniformes sur \(\{0,\ldots,4\}\), dont la moyenne théorique vaut 2 et l’écart-type \(\sqrt{2}\). Cela suggère une faible contribution axonale finale dans cette condition. Les statistiques agrégées ne suffisent toutefois pas à établir si cette situation vient des gradients à la borne zéro, de la projection entière ou d’une autre propriété de l’optimisation ; il faudrait examiner l’historique des délais et les gradients des runs concernés.

## Traçabilité et reproduction

Le point d’entrée commun est `experiments/train.py`, avec les hyperparamètres de `configs/perf_HAR.py`. Chaque run produit une configuration, les courbes `train_res.csv` et `val_res.csv`, les checkpoints `best.pth` et `last.pth`, ainsi qu’un résumé `final_test.json`. Pour les couches récurrentes, les fichiers `init_delays.npz`, `final_delays.npz` et, sur demande, `delay_history/` permettent de suivre les délais. Les délais finaux sont extraits après rechargement du meilleur checkpoint, qui n’est pas nécessairement celui de la dernière époque.

Les lanceurs présents dans le dépôt permettent de reconstruire les familles d’expériences :

| Expérience | Script ou option |
|---|---|
| Trois modèles feedforward, graines 0/1/2 | `experiments/scripts/test_HAR_feedforward.sh` |
| Complément des runs feedforward et relance de l’hybride récurrent | `experiments/scripts/finish_HAR.py` |
| Hybride à partie axonale nulle et gelée | `experiments/scripts/test_HAR_hybrid_zero.sh` |
| Hybride sans décalages synaptiques | `--hybrid-max-synaptic-delay 0` dans `experiments/train.py` |
| Variation de l’initialisation axonale | `experiments/scripts/test_HAR_hybrid_std.sh` |
| Historique comparé des délais récurrents | `experiments/scripts/test_HAR_delay_history.sh` |

Pour couvrir les quatre initialisations des slides avec le lanceur actuel, il faut explicitement étendre sa liste par défaut, qui ne contient que 0, 1 et 2 :

```bash
STDS="0 1 2 3" bash experiments/scripts/test_HAR_hybrid_std.sh
```

Pour l’ablation sans décalages, une commande explicite est :

```bash
python experiments/train.py --dataset har \
  --model SNN_recurrent_hybrid_delays --seeds 0,1,2 \
  --epochs 100 --init-rec-delay half_normal --delay-std-init 3 \
  --hybrid-max-synaptic-delay 0 --save-delay-history
```

Ces commandes correspondent aux conditions expérimentales décrites ; leur correspondance exacte avec chaque agrégat du PDF reste à établir à partir des chemins des runs et de leurs configurations archivées. En particulier, `finish_HAR.py` indique qu’il relance l’hybride récurrent après une correction de gradient. Il faut donc identifier la version du code associée aux mesures avant de mélanger des exécutions historiques et récentes.

## Portée des conclusions et vérifications éditoriales

Les expériences montrent un intérêt de la formulation hybride en feedforward à budget de paramètres appris constant, tandis que son gain n’apparaît pas dans la comparaison récurrente. Les deux ablations sont cohérentes avec les cas limites attendus de la paramétrisation. Enfin, les initialisations non nulles étudiées donnent des performances proches malgré des distributions de délais différentes ; l’initialisation exactement nulle constitue un cas distinct et moins performant.

La portée de ces conclusions reste celle du protocole employé : trois répétitions pour le balayage d’initialisation, absence de test statistique rapporté, découpage par fenêtres et réutilisation de l’évaluation pour choisir le checkpoint. Elles ne démontrent ni une généralisation à des sujets inconnus ni une équivalence statistique entre modèles.

Avant intégration définitive au rapport, trois points du support doivent être harmonisés à partir des données sources :

- L’axonal récurrent est donné à 82,896 ± 0,133 % en diapositive 6, mais à 82,896 ± 0,162 % en diapositive 12. Il faut vérifier les runs inclus et la convention de calcul de l’écart-type.
- L’hybride récurrent global atteint 82,855 ± 0,537 %, tandis que la condition d’initialisation 3 atteint 82,558 ± 0,614 %. Ces agrégats sont conservés séparément ici ; leur provenance ne doit pas être supposée identique.
- La nature des répétitions de l’ablation fixe doit être précisée : graines d’apprentissage, graines des décalages, ou variation des deux. Le lanceur actuel varie par défaut uniquement les décalages hybrides et n’exécute la référence synaptique que si `INCLUDE_SYNAPTIC=1`.

Sources locales : `docs/SNN-etude-de-cas.pdf`, diapositives 5–18 ; `configs/perf_HAR.py` ; `src/delrec/datasets/wisdm.py` et `loaders.py` ; `src/delrec/networks.py` ; `src/delrec/delay_layers.py` ; `src/delrec/training/har.py` ; `src/delrec/utils.py` ; `experiments/train.py` ; lanceurs de `experiments/scripts/`.
