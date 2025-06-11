from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold, GridSearchCV


from typing import Sequence, Dict, Any

def constructPipeline(regressor, preprocessing, target_transform):

    reg = regressor

    if preprocessing:

        reg = Pipeline([
            ("preprocessing", preprocessing),
            ("regressor", regressor)
        ])
    
    if target_transform:

        reg = TransformedTargetRegressor(
            regressor=reg,
            transformer=target_transform
        )
    
    return reg

def trainAndValidate(
        X,
        y,
        regressor,
        preprocessing,
        target_transform
    ):

    reg = constructPipeline(regressor, preprocessing, target_transform)

    kf = KFold(5, shuffle=True, random_state=42)

    scores = cross_validate(
        reg, 
        X, 
        y, 
        scoring=[
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
            "r2"
            ],
        cv=kf,
        n_jobs=-1,
        )
    
    return scores

def gridSearch(
        regressor,
        preprocessing,
        target_transform,
        param_grid: Dict[str, Sequence[Any]],
) -> GridSearchCV:
    
    reg = constructPipeline(regressor, preprocessing, target_transform)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=reg,
        param_grid=param_grid,
        scoring=[
            "neg_root_mean_squared_error",
            "neg_mean_absolute_error",
            "r2"],
        cv=kf,
        refit="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )

    return grid_search

