import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def optimize_decision_tree(train_x, train_y):
    # 定义决策树超参数搜索空间
    param_dist = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['log2', 'sqrt', None],  # Updated to include valid options
    }

    # 创建决策树模型
    decision_tree = DecisionTreeRegressor(random_state=1)

    # 使用RandomizedSearchCV进行随机搜索
    random_search = RandomizedSearchCV(
        decision_tree, 
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
    print("Best Model Performance(Negative value of average mean squared error):", random_search.best_score_)

    return best_model
