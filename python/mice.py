#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import statsmodels.api as sm
import statsmodels.imputation.mice as mice
import pandas as pd
import numpy as np


data = pd.DataFrame({
        "y":[1,0,3,9,np.nan,2,8,3,4],
        "x1":[0,0,30,50,np.nan,20,7,2,2],
        "x2":[8,7,np.nan,5,2,4,np.nan,4,2],
        "x3":[7,4,3,2,4,6,np.nan,np.nan,4],
        #"x4":["a","b",np.nan,"b","b","b","a","b","a"]
        })

#data = pd.DataFrame({
#        "x3":["a","b",np.nan,"b","b","b","a","b","a"],
#        "x4":["a","b",np.nan,"b","b","b","a","b","a"]
#        })



imp = None

# mice初期化
imp = mice.MICEData(data, 
                    perturbation_method="gaussian",  # or boot
                    k_pmm=20,
                    history_callback=None
                    )
# _initial_imputationで初期値を入れている模様

# miceの設定
imp.set_imputer("y", formula='x1 + x2', model_class=None,
                    init_kwds=None, fit_kwds=None, predict_kwds=None,
                    k_pmm=20, perturbation_method=None, regularized=False)

imp.set_imputer("x1", formula='y', model_class=None,
                    init_kwds=None, fit_kwds=None, predict_kwds=None,
                    k_pmm=20, perturbation_method=None, regularized=False)

# 設定された値一覧
imp.conditional_formula
imp.model_class
imp.perturbation_method
imp.regularized
imp.k_pmm
imp.init_kwds
imp.fit_kwds
imp.predict_kwds

# 内部データ
imp.data
imp._cycle_order

# mice実行
imp.update_all(n_iter=10)

#結果
imp.models
imp.models["y"].fit().summary()
imp.models["x1"].fit().summary()
imp.models["x2"].fit().summary()
imp.models["x3"].fit().summary()

imp.results
imp.results["y"].summary()
imp.results["x1"].summary()
imp.results["x2"].summary()
imp.results["x3"].summary()

imp.results["y"].params
imp.results["y"].cov_params()

imp.params

imp.data





# 以下は謎の処理
fml = 'y ~ x1 + x2 + x3'
mice_ins = mice.MICE(fml, sm.OLS, imp)
results = mice_ins.fit(10, 10)
results.summary()
results.predict(exog=imp.data.loc[:,["x1","x2","x3"]], transform=False)

results.model




# 一般的なstatsmodelsの利用方法(線形回帰)
data = pd.DataFrame({
        "y":[1,3,9,2,8,3,4],
        "x1":[0,30,50,20,7,2,2],
        "x2":[8,7,5,2,4,4,2],
        "x3":[7,4,3,2,4,6,4],
        "x4":["a","b","a","b","b","b","a"]
        })
res = sm.OLS(endog=data['y'],exog=data.loc[:,["x1","x2","x3","x4"]]).fit()
res.summary()
res.predict(exog=data.loc[:,["x1","x2","x3"]])
