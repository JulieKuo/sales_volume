import pandas as pd
import numpy as np
from pyod.models.knn import KNN
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.stats import skew 
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import cross_val_score
from traceback import format_exc
from log_config import Log
import warnings, sys, base64, json, os, pickle
warnings.filterwarnings("ignore")


log = Log()



class Model():
    def __init__(self, model_path, log_path):
        self.model_path = model_path
        self.input_path = os.path.join(model_path, "parser.csv")
        

        global logging
        logging = log.set_log(filepath = log_path, level = 2, freq = "D", interval = 50)



    def data_preprocess(self):
        logging.info("========== Data preprocess. ==========")


        '''Load data'''
        df = pd.read_csv(self.input_path)
        df["week"] = pd.to_datetime(df["week"])
        df = df.set_index(["week"])


        '''Missing Value'''
        df = df.fillna(0) # 無銷售量資料代表該週無交易 = 0


        '''outlier & expotential'''
        df_outlier = pd.DataFrame(index = df.index)
        for col, value in df.items():
            clf = KNN()
            clf.fit(df[[col]])  # 这里X_train的维度是 [n_samples, n_features]，必须是数值型，不能包含缺失值

            ## 返回训练数据的标签和分值
            outlier = clf.labels_  # (0表示正常值，1表示异常值)
            max_ = df[col].iloc[~outlier.astype(bool)].max()
            min_ = df[col].iloc[~outlier.astype(bool)].min()

            value = [max_ if i > max_ else i for i in value]
            value = [min_ if i < min_ else i for i in value]

            df_outlier[col] = value


        '''每四週累加，單位週->月'''
        df_cum = df_outlier.rolling(4).sum()
        df_cum = df_cum.dropna()


        return df_cum



    def RG_to_Class(self, series):
        logging.debug("Regression to Classification.")


        '''切分 (連續變數轉分類預測)'''
        # 最小值和最大值的十進位數
        digit1 = len(str(int(series["qty"].max()))) + 1
        digit2 = len(str(int(series["qty"].min()))) - 1


        # 產生區間
        bins = []
        for i in range(digit2, digit1):
            if i >= 4: # 超過10000，每一個進位區間中多切四份，避免區間過大
                bins1 = np.linspace(10**i, 10**(i+1), 5)#+(i-4)*2
                if i > 4:
                    bins1 = np.delete(bins1, 0) # 刪除重複的數字
                bins.extend(bins1)
            else:
                bins.append(10**i)
                
        bins[0] = bins[0] - 1


        # 切割
        series["cut"] = pd.cut(series["qty"], bins) # 方便之後看label的區間是多少
        series["label"] = pd.cut(series["qty"], bins, labels = list(range(len(bins)-1))).astype(int)


        # 儲存各類別對應的區間
        cat = series[["cut", "label"]].drop_duplicates().sort_values("label")
        cat = cat.set_index("label")
        series = series.drop(["cut"], axis = 1)


        return series, cat



    def Feature_engineering(self, series, model_col_path):
        logging.debug("Feature engineering.")


        '''特徵生成'''
        df1 = series.drop(["qty"], axis = 1)
        # 滯後特徵
        for i in range(4, 13):
            df1[f"label_lag_{i}"] = df1["label"].shift(i)

        # 時間特徵
        df1["year"] = df1.index.year
        df1["month"] = df1.index.month
        df1["quarter"] = df1.index.quarter
        df1["week"] = df1.index.week

        df2 = df1.dropna()
        df2 = df2.reset_index(drop = True)


        '''Label encoding'''
        ord_enc = OrdinalEncoder()
        df2['year'] = ord_enc.fit_transform(df2[['year']]).astype(int)#對年做排序


        '''Train test split'''
        X_train, X_test, y_train, y_test = train_test_split(df2.drop(["label"], axis = 1), df2[["label"]], test_size=0.2, shuffle = False)


        '''Resample'''
        # 刪除數量少於2的類別
        counts = y_train["label"].value_counts() 
        drop_label = counts[counts < 2].index.to_list()

        y_train = y_train.query("label not in @drop_label")
        X_train = X_train.loc[y_train.index]


        # 刪除test有， train沒有的data
        train_label = y_train["label"].unique()
        y_test = y_test.query("label in @train_label")
        X_test = X_test.loc[y_test.index]


        # 檢查類型中最少的數量是否比ADASYN模型所默認的參數還小
        if drop_label != []:
            counts = counts.drop(drop_label)
            n_neighbors = min(counts.min()-1, 5)
        else:
            n_neighbors = counts.min()-1


        # resample
        oversample = SMOTE(random_state=99, k_neighbors = n_neighbors)
        X_train, y_train = oversample.fit_resample(X_train, y_train)


        '''Skew'''
        skewness = X_train.select_dtypes(float).apply(lambda X: skew(X)).sort_values(ascending=False)
        skewness = pd.DataFrame({'Feature' : skewness.index, 'Skew' : skewness.values})
        skewness = skewness.query("(Skew > 0.75) | (Skew < -0.75)")
        skewness = skewness.reset_index(drop = True)
        skewness.to_csv(os.path.join(model_col_path, "skew_feat.csv"), index = False)

        if len(skewness != 0):
            X_pt = PowerTransformer(method = 'yeo-johnson')
            X_train[skewness["Feature"]] = X_pt.fit_transform(X_train[skewness["Feature"]])
            X_test[skewness["Feature"]] = X_pt.transform(X_test[skewness["Feature"]])

            pickle.dump(X_pt, open(os.path.join(model_col_path, "skew.pickle"), "wb"))


        '''Scaling'''
        X_scaler = MinMaxScaler()
        X_train = pd.DataFrame(X_scaler.fit_transform(X_train), columns = X_scaler.feature_names_in_)
        X_test = pd.DataFrame(X_scaler.transform(X_test), columns = X_scaler.feature_names_in_)

        pickle.dump(X_scaler, open(os.path.join(model_col_path, "scaler.pickle"), "wb"))


        return X_train, X_test, y_train, y_test



    def modeling(self, X_train, X_test, y_train, y_test, col, model_col_path):
        logging.debug("Modeling.")
        

        '''Train'''
        random_state = None
        models = {
            "Logistic": LogisticRegression(),
            "Bayes (Gaussian)": GaussianNB(),
            "Bayes (Complement)": ComplementNB(), # for imbalance data (X cannot be negative)
            "KNN": KNeighborsClassifier(),
            "SVC": SVC(probability = True),
            "Neural Network": MLPClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state = random_state),
            "Random Forest": RandomForestClassifier(random_state = random_state, class_weight="balanced"),
            "Gradient Boost": GradientBoostingClassifier(),
            "LightGBM": LGBMClassifier(random_state = random_state), # feature name should be number
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            logging.debug(f"{name} trained.")


        '''Evaluate models'''
        score = []
        flag = 1
        cv_flag = 0
        for key, model in models.items():
            pred_train = model.predict(X_train)
            pred_test  = model.predict(X_test)

            acc_train  = accuracy_score(y_train, pred_train).round(2)
            acc_test   = accuracy_score(y_test, pred_test).round(2)

            recall_train  = recall_score(y_train, pred_train, average = 'weighted').round(2)
            recall_test   = recall_score(y_test, pred_test, average = 'weighted').round(2)

            precision_train  = precision_score(y_train, pred_train, average = 'weighted').round(2)
            precision_test   = precision_score(y_test, pred_test, average = 'weighted').round(2)

            f1_train = f1_score(y_train, pred_train, average = 'weighted').round(2)
            f1_test = f1_score(y_test, pred_test, average = 'weighted').round(2)
            f1_macro = f1_score(y_test, pred_test, average = 'macro').round(2)
            
            if flag == 1:
                col1 = [acc_train, acc_test, recall_train, recall_test, precision_train, precision_test, f1_train, f1_test, f1_macro]
                col2 = ["Accuracy_train", "Accuracy_test", "Recall_train", "Recall_test", "Precision_train", "Precision_test", "f1_train", "f1_test", "f1_test_macro"]
            else:
                col1 = [acc_test, recall_test, precision_test, f1_test]
                col2 = ["Accuracy", "Recall", "Precision", "f1"]

            if cv_flag:
                cv_scores = cross_val_score(model, X_train, y_train, cv = 3, scoring = 'accuracy')
                cv_score = cv_scores.mean().round(2)
                col1 += [cv_score]
                col2 += ["f1_cv"]

            score.append(col1)
            

        index  = [i.lstrip() for i in models.keys()]
        score  = pd.DataFrame(score, index = index, columns = col2)

        score1 = score.sort_values(["Accuracy_test", "f1_test_macro", "f1_test"], ascending = False)
        best_model_name = score1.index[0]
        # dump(models[best_model_name], os.path.join(model_col_path, "model.joblib"))
        pickle.dump(models[best_model_name], open(os.path.join(model_col_path, "model.pickle"), "wb"))
        


        score1 = score.loc[[best_model_name]].reset_index()
        score1 = score1.rename(columns = {"index": "model"}, index = {0: col})


        return score1



    def main(self):
        try:
            df_cum = self.data_preprocess()

            logging.info("========== Train each series. ==========")
            
            total_series = {}
            total_cat = {}
            scores =  pd.DataFrame()
            
            for col in df_cum.columns:
                logging.info(f"---------- {col} ----------")

                model_col_path = os.path.join(self.model_path, f"model/{col}")
                if not os.path.isdir(model_col_path): # create dir if dir doesn't exist
                    os.makedirs(model_col_path)

                series = df_cum[[col]].copy()
                series.columns = ["qty"]
                
                series, cat = self.RG_to_Class(series)


                total_series[col] = series
                total_cat[col] = cat

                X_train, X_test, y_train, y_test = self.Feature_engineering(series, model_col_path)

                score = self.modeling(X_train, X_test, y_train, y_test, col, model_col_path)
                scores = pd.concat([scores, score])
                

            scores.to_csv(os.path.join(self.model_path, 'scores.csv'))
            pickle.dump(total_series, open(os.path.join(self.model_path, 'model/series.pickle'), "wb"))
            pickle.dump(total_cat, open(os.path.join(self.model_path, 'model/categories.pickle'), "wb"))

            acc = {"accuracy": scores["Accuracy_test"].max()}
            acc_path = os.path.join(self.model_path, "result.json")
            with open(acc_path, 'w', newline='') as file:
                json.dump(acc, file)
                

            logging.info("========== Finish. ==========")

        except:
            logging.error(format_exc())

        finally:
            log.shutdown()


        return scores



if __name__ == '__main__':

    if len(sys.argv) > 1: 
        input_ = sys.argv[1]
        input_ = base64.b64decode(input_).decode('utf-8')

        input_ = json.loads(input_)
    else:
        print("Input parameter error.")


    model_path = input_["model_path"]
    log_path = input_["log_path"]

    
    model = Model(model_path)
    scores = model.main()