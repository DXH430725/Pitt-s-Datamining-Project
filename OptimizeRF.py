import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def optimize_random_forest(train_x, train_y):
    # 定义Random Forest超参数搜索空间
    param_dist = {
        'n_estimators': randint(10, 200),
        'max_features': ['log2', 'sqrt', None],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'bootstrap': [True, False]
    }

    # 创建Random Forest模型
    random_forest = RandomForestRegressor(random_state=1)

    # 使用RandomizedSearchCV进行随机搜索
    random_search = RandomizedSearchCV(
        random_forest, 
        param_distributions=param_dist, 
        n_iter=10,  # 设置搜索的迭代次数
        scoring='neg_mean_squared_error',  # 根据任务选择适当的评估指标
        cv=5,  # 交叉验证的折数
        random_state=42
    )

    # 执行搜索
    random_search.fit(train_x, train_y)

    # 输出最佳参数
    best_params = random_search.best_params_
    print("Best Parameters:", best_params)

    # 输出最佳模型的性能
    best_model = random_search.best_estimator_
    print("Best Model Performance (Negative value of average mean squared error):", random_search.best_score_)

    return best_model
