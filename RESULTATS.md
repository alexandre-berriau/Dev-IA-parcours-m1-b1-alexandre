# Modèle d'annulation v2 — Résultats

## Démarche

1. Je charge les 78 703 réservations en une requête

2. Je mets de côté les 4 derniers mois d'arrivée comme test. 

3. Un premier modèle sur toutes les colonnes m'a sorti 100 % de réussite. Pas bon signe, cela montre qu'il y a une donnée qui provoque une fuite. Une fois retirée, j'ai rejoué la recette de la v1 sur mon seul train pour avoir une référence comparable : F1 0,7005.

4. Ma v2 est un pipeline qui prend les données brutes, crée 4 variables, encode, impute et classe. J'ai comparé trois familles, la forêt gagne. Je l'ai réglée en deux grilles et j'ai choisi le seuil sur les 4 derniers mois du train.

5. Le test n'a servi qu'une fois, à la fin.

## Données

| | Période d'arrivée | Réservations | Taux d'annulation |
|---|---|---|---|
| Entraînement | 2015-07-01 → 2016-08-31 | 58 792 | 35,6 % |
| Test | 2016-09-01 → 2016-12-31 | 19 911 | 38,0 % |

## Fuite de données

Mon premier modèle a donné 1,0000 d'exactitude. Un score parfait
n'existe pas, j'ai cherché l'erreur.

Les importances l'ont désigné : trois colonnes issues de `reservation_status`
pesaient 65 % du modèle à elles seules. Le `crosstab` a confirmé.

| `reservation_status` | maintenue | annulée |
|---|---|---|
| Check-Out | 37 883 | 0 |
| Canceled | 0 | 20 206 |
| No-Show | 0 | 703 |

Ça donne directement la cible donc fausse l'entraînement. Et j'ai 3 autres colonnes dans ce cas, qui n'existeraient qu'après la réservation.

| Colonne | Pourquoi |
|---|---|
| `reservation_status` | équivaut exactement à la cible |
| `reservation_status_date` | date du statut final, donc postérieure |
| `assigned_room_type` | 4,3 % d'annulation quand la chambre change, 41,1 % sinon. On n'attribue pas de chambre à quelqu'un qui ne vient pas |
| `booking_changes` | 39,2 % à zéro modification, 13,7 % dès la première. Au moment de prédire, cette colonne vaut zéro pour tout le monde |

J'ai aussi écarté `arrival_date_week_number`, pour une autre raison : elle faisait
perdre 3 points de F1 et portait l'écart train/validation de 0,17 à 0,21. Le modèle
apprenait des semaines de 2015 et 2016 qui ne reviennent pas.

## Variables nouvelles

| Variable | Calcul | Justification métier |
|---|---|---|
| `duree_sejour` | `stays_in_weekend_nights + stays_in_week_nights` | Une nuit et deux semaines n'engagent pas pareil. |
| `taille_groupe` | `adults + children + babies` | Une chambre pour une personne s'annule sur un coup de tête contrairement à une famille |
| `avec_enfants` | `(children + babies) > 0` | Un séjour calé sur les vacances scolaires |
| `taux_annulation_passe` | `previous_cancellations / (previous_cancellations + previous_bookings_not_canceled)`, 0 si client inconnu | Deux annulations sur trois, ce n'est pas dix réservations honorées. La base donnait les compteurs, pas le rapport |

Ces quatre variables font passer le F1 de 0,7411 à 0,7460 en forêt... pas extraordinaire mais cela a le mérite de l'améliorer. Ce qui joue réellement c'est le changement de famille.

## Validation et réglage

Validation croisée 5 plis sur le train, métrique F1.

| famille | F1 | précision | rappel | écart train/validation |
|---|---|---|---|---|
| Forêt aléatoire | 0,7485 | 0,8523 | 0,6853 | 0,1723 |
| Boosting (celle de la v1) | 0,7121 | 0,7893 | 0,6902 | 0,1682 |
| Régression logistique | 0,6679 | 0,7682 | 0,6230 | 0,0827 |

J'ai réglé deux hyperparamètres de la forêt : `min_samples_leaf` et `max_features`.

| `min_samples_leaf` | `max_features` | train | validation | écart |
|---|---|---|---|---|
| 1 | 0.5 | 0,9915 | 0,7281 | 0,2634 |
| 2 | sqrt | 0,9209 | 0,7485 | 0,1723 |
| 20 | 0.5 | 0,8650 | 0,7624 | 0,1026 |
| **100** | **0.7** | 0,8303 | **0,7679** | **0,0624** |
| 250 | 0.5 | 0,8083 | 0,7564 | 0,0519 |

La première ligne dit tout. À `min_samples_leaf=1`, le modèle fait 0,9915 sur ses
propres données et c'est le pire en validation. Il a appris les 58 792 lignes.

Le réglage gagne 2 points de F1, mais surtout il divise l'écart par presque trois.
C'est ce qui compte pour tenir sur des mois jamais vus.

J'ai aussi réglé le boosting, pour vérifier mon choix de famille. Il plafonne à
0,7327. La forêt reste devant.

En `TimeSeriesSplit`, la forêt réglée donne 0,7416 au lieu de 0,7679.

## Choix de la métrique

Une annulation ratée, c'est une chambre qui part vide. 310 € en moyenne sur le test.

Une annulation prédite à tort, c'est une chambre remise en vente alors que le
client vient. Il faut le reloger, et il ne revient pas.

Dans les deux cas cela a un coût réel pour l'hôtel, aucune ne domine. Je prends le F1, qui tombe dès que l'un des deux côtés se dégrade.

J'ai chiffré pour vérifier. En comptant une fausse alerte au double d'une annulation
ratée, la v2 coûte 1 407 000 € sur quatre mois contre 1 528 000 € pour la v1. Elle
rattrape 997 annulations de plus pour 303 fausses alertes. Il faudrait qu'une fausse
alerte coûte plus de 3,3 fois une annulation ratée pour que la v1 repasse devant.
Reloger un client ne coûte pas trois nuitées et demie.

### Le seuil

J'ai écrit une méthode dans `04_evaluation` qui balaie les seuils et garde le
meilleur F1. Je la fais tourner sur les 4 derniers mois du train, le modèle
n'apprenant pas dessus. Le test n'a jamais servi à ça. Elle donne 0,30 pour la v2
et 0,20 pour la v1.

## Verdict : v1 rejouée contre v2, sur le test

Une seule évaluation, 19 911 réservations, chaque modèle à son seuil.

| | v1 (recette rejouée) | v2 |
|---|---|---|
| **F1** | 0,6756 | **0,7297** |
| Précision | 0,8095 | 0,8150 |
| Rappel | 0,5798 | 0,6605 |
| Exactitude | 0,7884 | 0,8139 |
| ROC AUC | 0,8669 | 0,8797 |
| Seuil | 0,20 | 0,30 |
| Entraînement | 4,0 s | 1,9 s |
| Taille | 0,7 Mo | 6,4 Mo |

Au seuil 0,5 par défaut, l'écart est encore plus net : 0,5752 contre 0,6766.

**v1, seuil 0,20**

| | prédit : maintenue | prédit : annulée |
|---|---|---|
| réel : maintenue | 11 308 | 1 033 |
| réel : annulée | 3 181 | 4 389 |

**v2, seuil 0,30**

| | prédit : maintenue | prédit : annulée |
|---|---|---|
| réel : maintenue | 11 206 | 1 135 |
| réel : annulée | 2 570 | 5 000 |

La v2 rattrape 611 annulations de plus, pour 102 fausses alertes de plus.

### Les deux chutent sur le test

| | validation croisée | test | écart |
|---|---|---|---|
| v1 | 0,7005 | 0,5752 | −0,125 |
| v2 | 0,7679 | 0,6766 | −0,091 |

Chiffres au seuil 0,5, pour comparer ce qui est comparable.

Le test couvre l'automne 2016, que ni l'un ni l'autre n'a vu, avec 38,0 %
d'annulations contre 35,6 %. La validation croisée mélange les périodes, donc elle
surestime. Le `TimeSeriesSplit` l'avait annoncé avec ses 0,7416.

La v2 chute moins que la v1, 9 points contre 12,5. Le réglage l'a rendue plus stable.

## Recommandation

Je mets la v2 en service. Elle bat la v1 de 5,4 points de F1 sur le même test, elle
s'entraîne deux fois plus vite, et elle rattrape 611 annulations de plus sur quatre
mois.

Trois réserves.

Le rappel reste à 0,66. Un tiers des annulations passe encore à travers.

Les scores baissent dès qu'on change de saison. Il faut réentraîner régulièrement,
pas déployer une fois pour toutes. Ça coûte deux secondes.

La prédiction sert à relancer le client, pas à surbooker. À 81 % de précision, une
chambre sur cinq remise en vente l'aurait été à tort. C'est l'hypothèse derrière mon
seuil de 0,30. Pour surbooker vraiment, il faut remonter à 0,50 et rattraper moins
d'annulations.

## Reproduire

```bash
docker compose up -d
pip install -r requirements.txt
python train.py
```

`train.py` recharge les données, refait le découpage, entraîne la v2 et écrit
`model/cancellation_model_v2.joblib` et `model/model_info_v2.json`. Tous les modèles
ont `random_state=42`, les chiffres ci-dessus sont reproductibles à l'identique.

Le bloc final du script se réimporte sous le nom `train` avant d'appeler `main()`.
Sans ça, joblib enregistre `ajouter_variables` sous `__main__` et le modèle n'est
rechargeable que depuis `train.py`.
