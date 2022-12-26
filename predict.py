import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from dateutil.relativedelta import relativedelta
from traceback import format_exc
from log_config import Log
import warnings, datetime, sys, base64, json, os, pickle
warnings.filterwarnings("ignore")


log = Log()



class Predict():
    def __init__(self, start, end, model_path, output_path, log_path):
        self.start = start
        self.end = end
        self.model_path = model_path
        self.output_path = output_path


        global logging
        logging = log.set_log(filepath = log_path, level = 2, freq = "D", interval = 50)



    def predict(self, start, end, series, cat, col, results):
        '''Predict'''
        logging.debug("Predict.")


        # load model
        best_model = pickle.load(open(os.path.join(self.model_path, f'model/{col}/model.pickle'), "rb"))
        X_scaler = pickle.load(open(os.path.join(self.model_path, f'model/{col}/scaler.pickle'), "rb"))
        skewness = pd.read_csv(os.path.join(self.model_path, f'model/{col}/skew_feat.csv'))
        if len(skewness != 0):
            X_pt = pickle.load(open(os.path.join(self.model_path, f'model/{col}/skew.pickle'), "rb"))


        # 各類別的統計資料
        medain = series.groupby("label")["qty"].median()
        std = series.groupby("label")["qty"].std()


        # 抓出歷史的相關資料
        series["start"] = None
        series["end"] = None
        series["median"] = None
        series["random"] = None
        series["pred"] = None

        for i in range(len(series)):
            pred = series.iloc[i, 1]
            series.iloc[i, 2] = cat.loc[pred]['cut'].left
            series.iloc[i, 3] = cat.loc[pred]['cut'].right
            series.iloc[i, 4] = medain[pred]

            if np.isnan(std[pred]) or (std[pred] == 0):
                series.iloc[i, 5] = np.random.randint(5, 10)
            else:
                series.iloc[i, 5] = round(np.random.normal(scale = std[pred]*0.5))

            series.iloc[i, 6] = series.iloc[i, 4] + series.iloc[i, 5]
            if (series.iloc[i, 6] < series.iloc[i, 2]):
                series.iloc[i, 6] = series.iloc[i, 2]
            elif (series.iloc[i, 6] > series.iloc[i, 3]):
                series.iloc[i, 6] = series.iloc[i, 3]

        
        # 預測未來一年的資料
        start_pred = series.index[-1]
        pred_date = start_pred + relativedelta(weeks = 1)
        while pred_date < end:
            # 建立預測日期
            series.loc[pred_date] = None
            df1 = series.copy()


            #特徵生成
            ## 滯後特徵
            for i in range(4, 13):
                df1[f"label_lag_{i}"] = df1["label"].shift(i)

            ## 時間特徵
            df1["year"] = df1.index.year
            df1["month"] = df1.index.month
            df1["quarter"] = df1.index.quarter
            df1["week"] = df1.index.week


            # 特徵工程
            ## label-encoding
            ord_enc = OrdinalEncoder()
            df1[["year"]] = ord_enc.fit_transform(df1[["year"]])

            pred_X = df1.iloc[-1:, -13:]

            ## skew
            if len(skewness != 0):
                pred_X[skewness["Feature"]] = X_pt.transform(pred_X[skewness["Feature"]])
            ## scaling
            pred_X = pd.DataFrame(X_scaler.transform(pred_X), columns = X_scaler.feature_names_in_)


            # predict
            pred = best_model.predict(pred_X)


            # save predict
            series.iloc[-1, 1] = pred[0]
            series.iloc[-1, 2] = cat.loc[pred[0]]['cut'].left
            series.iloc[-1, 3] = cat.loc[pred[0]]['cut'].right
            series.iloc[-1, 4] = medain[pred[0]]

            if np.isnan(std[pred[0]]) or (std[pred[0]] == 0):
                series.iloc[-1, 5] = np.random.randint(5, 10)
            else:
                series.iloc[-1, 5] = round(np.random.normal(scale = std[pred[0]]*0.5))
            
            series.iloc[-1, 6] = series.iloc[-1, 4] + series.iloc[-1, 5]
            if (series.iloc[-1, 6] < series.iloc[-1, 2]):
                series.iloc[-1, 6] = series.iloc[-1, 2]
            elif (series.iloc[-1, 6] > series.iloc[-1, 3]):
                series.iloc[-1, 6] = series.iloc[-1, 3]

            pred_date = series.index[-1] + relativedelta(weeks = 1)

        
        # 抓出每個月最後一周 (即實際值最後一週往前推四週的累積)
        series["year"] = series.index.year
        series["month"] = series.index.month
        series_keep = series[["year", "month"]].drop_duplicates(keep = "last")
        series = series.loc[series_keep.index]
        

        # 抓出預測區間內的資料
        series1 = series[(series.index >= start) & (series.index <= end)]
        series1 = series1.drop(["year", "month"], axis = 1)
        series1.iloc[:, 1:] = series1.iloc[:, 1:].astype(int)
        
        # 按年分儲存
        series2 = series1[["pred"]]
        series2["year"] = series2.index.year
        g = series2.groupby("year")
        for year in g.size().index:
            result = g.get_group(year)
            result = result.drop("year", axis = 1)
            col = "其他" if col == "other" else col
            result.columns = [col]
            result = result.T
            result = result.reset_index()
            result.columns = ["series"] + [f"{i}m" for i in range(1, 13)]
            results[year] = pd.concat([results[year], result])


        return results



    def main(self):
        try:
            # 預測區間
            start = datetime.datetime(self.start, 1, 1)
            end = datetime.datetime(self.end, 12, 31)


            # load date
            logging.info("========== Load Data. ==========")
            total_series = pickle.load(open(os.path.join(self.model_path, 'model/series.pickle'), "rb"))
            total_cat = pickle.load(open(os.path.join(self.model_path, 'model/categories.pickle'), "rb"))


            logging.info("========== Predict each series. ==========")
            
            results = {i: pd.DataFrame() for i in range(start.year, end.year+1)}
            for col in total_series.keys():
                logging.info(f"---------- {col} ----------")
                series = total_series[col]
                cat = total_cat[col]

                results = self.predict(start, end, series, cat, col, results)
            

            if not os.path.isdir(self.output_path): # create dir if dir doesn't exist
                    os.makedirs(self.output_path)

            for i in results.keys():
                results[i].to_csv(os.path.join(self.output_path, f"{i}.csv"), index = False)
                

            logging.info("========== Finish. ==========")

        except:
            logging.error(format_exc())

        finally:
            log.shutdown()


        return results


if __name__ == '__main__':

    if len(sys.argv) > 1: 
        input_ = sys.argv[1]
        input_ = base64.b64decode(input_).decode('utf-8')

        input_ = json.loads(input_)
    else:
        print("Input parameter error.")


    start = int(input_["START"])
    end = int(input_["END"])
    model_path = input_["MODEL"]
    output_path = input_["OUTPUT"]
    log_path = input_["LOG"]

    
    predict = Predict(start, end, model_path, output_path, log_path)
    results = predict.main()