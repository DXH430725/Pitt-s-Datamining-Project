import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

def optimize_svm(train_x, train_y):
    # 定义SVM超参数搜索空间
    param_dist = {
        'C': uniform(0.1, 10),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'] + list(uniform(0.1, 1.0).rvs(5)),
        'degree': [2, 3, 4, 5],
    }

    # 创建SVM模型
    svm_model = SVR()

    # 使用RandomizedSearchCV进行随机搜索
    random_search = RandomizedSearchCV(
        svm_model, 
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
