- transformer avant LSTM: ?


- modèles
    - transformer normal
    - closed choice no router
    - closed choice router
    - ours
    - ours adaptive computation
    - moyenner loss avec T fixe

    - [o] avoir bon transformer avant de tester autre chose
        - [x] lancer opt
        - [ ] générations pire et meilleur config
    - [ ] before further ours
        - [ ] optimize D_router
        - [ ] k couches identité (en plus des n layers)
        - [ ] T max avant ACT
            - [ ] opt en loss cumulée et test au dernier t
    - [ ] analyses
        - [ ] does adding identity layers help?
        - [ ] ours vs transformer
        - [ ] Stats sur controller
            - [ ] ordre de déploiement couches
                - [ ] mat layer/T
            - [ ] appariement 
                - [ ] mat layer/layer2
            - [ ] are identity layers used? When?
        - [ ] validation pipeline
    
