import optuna
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer

optuna.logging.set_verbosity(optuna.logging.WARNING) #isto é para silenciar os avisos reccorrentes de cada tentativa individual

##esta primeira é o tuning apenas como temos na aula
#
#def tune_pipeline(pipeline, X, y, cv, scoring: str, search_space: dict, n_trials: int, random_state: int, n_jobs: int = 1):
#    def objective(trial): 
#        params = {}
#        for name, spec in search_space.items():
#            if spec["type"] == "int":
#                params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False)) #aqui está a escolher quais os valores que vai testar
#            elif spec["type"] == "float":
#                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
#            elif spec["type"] == "categorical":
#                params[name] = trial.suggest_categorical(name, spec["choices"])
#        
#        candidate = clone(pipeline).set_params(**params) #aqui cria umm clone do modelo e limpa-o para ter a certeza que nao ha data leakage e depois aplica os valores indicados
#        scores = cross_val_score(candidate, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs) #ver o quao bem o modelo deu perform e devolve ao optuna
#        trial.set_user_attr("std", float(scores.std(ddof=1)))
#        return scores.mean()
#
#    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)) #indica que queremos a maior pontuacao possivel na metrica que foi escolhida 
#    study.optimize(objective, n_trials=n_trials) #executa o loop  objective o número de vezes que definimos nos trials
#    
#    best_pipeline = clone(pipeline).set_params(**study.best_params) #Cria e devolve um pipeline novo (ainda não treinado) configurado com a melhor combinação de hiperparâmetros encontrada
#    return best_pipeline, study
#
##esta é o tuning com outras opcoes para alem do TPE do optuna podemos ainda escolher se queremos usar Random Search, CMA-ES ou Grid Search

def tune_pipeline(pipeline, X, y, groups, cv, scoring: str, search_space: dict, n_trials: int, random_state: int, n_jobs: int = 1, sampler_type: str = "tpe"):
    """
    Executa o estudo do Optuna permitindo escolher entre 4 samplers:
- 'tpe': Tree-structured Parzen Estimator (Bayesiano)
- 'random': Random Search (Aleatório)
- 'cmaes': CMA-ES (Evolutivo)
- 'grid': Grid Search (Busca em Grelha)
    """
    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            param_key = name if name.startswith("model__") else f"model__{name}"
            if spec["type"] == "int":
                params[param_key] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "float":
                params[param_key] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "categorical":
                params[param_key] = trial.suggest_categorical(name, spec["choices"])
        
        candidate = clone(pipeline).set_params(**params)
        scores = cross_val_score(candidate, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=n_jobs)
        trial.set_user_attr("std", float(scores.std(ddof=1)))
        return scores.mean()

    sampler_type = sampler_type.lower()
    
    if sampler_type == "random":
        sampler = optuna.samplers.RandomSampler(seed=random_state)
    elif sampler_type == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(seed=random_state)
    elif sampler_type == "grid":

        grid_search_space = {}
        for name, spec in search_space.items():
            if spec["type"] == "categorical":
                grid_search_space[name] = spec["choices"]
            elif spec["type"] == "int":
                step = spec.get("step", 1)
                grid_search_space[name] = list(range(spec["low"], spec["high"] + 1, step))
            elif spec["type"] == "float":
                grid_search_space[name] = list(np.linspace(spec["low"], spec["high"], num=5))
        sampler = optuna.samplers.GridSampler(search_space=grid_search_space, seed=random_state)
    else: # Por padrão usa TPE
        sampler = optuna.samplers.TPESampler(seed=random_state)

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    best_params_mapped = {
        (k if k.startswith("model__") else f"model__{k}"): v 
        for k, v in study.best_params.items()
    }
    best_pipeline = clone(pipeline).set_params(**best_params_mapped)
    return best_pipeline, study

def nested_cross_validate(pipeline, X, y, groups, outer_cv, inner_cv, scoring: str, search_space: dict, n_trials: int, random_state: int, n_jobs: int = 1, sampler_type: str = "tpe"):
    scorer = get_scorer(scoring)
    rows = []
    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype) #neste pedaço é cross validation em que ele faz o tuning so sobre o train e
    
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y, groups=groups), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        groups_train = groups.iloc[train_idx]

        best_pipeline, study = tune_pipeline(
            pipeline, X_train, y_train, groups_train, inner_cv, scoring,
            search_space, n_trials, random_state, n_jobs, sampler_type=sampler_type
        )
        best_pipeline.fit(X_train, y_train)

        train_score = scorer(best_pipeline, X_train, y_train)
        val_score = scorer(best_pipeline, X_val, y_val)
        y_oof[val_idx] = best_pipeline.predict(X_val)
        
        rows.append({
            "fold": fold, 
            "train": train_score, 
            "validation": val_score,
            "gap": train_score - val_score, 
            "inner_best": study.best_value,
            **study.best_params
        })
        
    return pd.DataFrame(rows), y_oof