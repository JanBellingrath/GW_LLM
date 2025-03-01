- transformer avant LSTM: ?


- modèles
    - transformer normal
    - closed choice no router
    - closed choice router
    - ours
    - ours adaptive computation
    - moyenner loss avec T fixe

    - [x] avoir bon transformer avant de tester autre chose
        - [x] lancer opt
        - [x] générations pire et meilleur config
    - [ ] before further ours
        - [x] optimize D_router
        - [x] k couches identité (en plus des n layers)
        - [x] T max avant ACT
            - [x] opt en loss cumulée et test au dernier t
        - [ ] routing for each token
            - [x] iterate over tokens, 
            - [x] hide future to router
            - [x] fix order analysis with change of shape
            - [x] TODO: find parts of the batch where expert is selected 
            - [x] except if targets is None (i.e. inference-time)
    - [ ] analyses
        - [ ] does adding identity layers help?
        - [ ] ours vs transformer
        -> [ ] check nb params, et same for each timestep?
        - [ ] Stats sur controller
            - [ ] ordre de déploiement couches
                - [ ] mat layer/T
            - [ ] appariement 
                - [ ] mat layer/layer2
            - [ ] are identity layers used? When?
        - [ ] validation pipeline
    - [ ] augmenter taille contexte ?
    - [ ] régularisation des experts ?
    - [ ] Router architecture
        - [ ] LSTM vs lineaire ?

    
- To discuss with Rufin
    - [ ] même routage pour tous les tokens
    - [ ] petit contexte (256)
    - [ ] tokenizer interdit

    - [ ] continuer premier modèle arithmétique, rufin chaud pour workshop
    - [x] entraîner plus comme BERT, prédire un token à la fois
        - [x] check how it is trained now
            - [x] BERT training: hide or corrupt token with probability (multiple tokens can be selected)


Todo Rufin:
    - [ ] transformer comparison
    - [ ] préentrainer couches, capable de refaire archi? Déviation de l'archi dans certains cas ?
    - [ ] arithmetic chaining

- quelle dim input routeur chez mixtral ?
- peut sauver comput en appelant qu'une fois transformer sur tous tokens si au moins un du batch a besoin de l'expert

Discussion Rufin:
    - Routeur
        - Besoin RNN ?
        - Besoin d'envoyer tous les tokens ? Mixtral gate input: (batch * sequence_length, hidden_dim) (mais après contextualisation car MoE que pour MLP)

Todo aujourd'hui:
    - [ ] arithmetic
        - [ ] tester si fonctionne toujours
    - [ ] tester si moins d'utilisation mémoire
    - [ ] lancer slow with simple router


- [ ] dire au routeur où on est (pos ou scalar), ou alors donner toute la séquence
- [ ] correlation enlever diagonale
- [ ] matrice de fréquence de sélection conjointe d'experts
- [ ] plots de routage instance-wise
- [ ] augmenter nb de couches et pas de temps, pas forcément embd pour limiter transformer, et voir si toujours moins bon que trans