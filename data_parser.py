from log_config import Log
import pandas as pd
from traceback import format_exc
import os, sys, base64, json


log = Log()



class Parser():
    def __init__(self, model_path, log_path):
        self.input_path = os.path.join(model_path, "data.csv")
        self.output_path = os.path.join(model_path, "parser.csv")
        self.origin_path = os.path.join(model_path, "..\\origin_data.csv")  
        

        global logging
        logging = log.set_log(filepath = log_path, level = 2, freq = "D", interval = 50)


        
    def main(self):
        try:
            logging.info("========== Data parsing. ==========")
            
            # 載入歷史資料
            df0 = pd.read_csv(self.origin_path)
            df0["week"] = pd.to_datetime(df0["week"])
            df0 = df0.set_index("week")
            
            # 載入新資料
            df = pd.read_csv(self.input_path)
            df = df.drop_duplicates()

            # 合併GIGE、USB2.0和USB3.0，確認產品名稱
            df["產品系列"] = df["產品系列"].str.upper()
            series = ["GIGE", "USB2.0", "USB3.0", "其他", "UNKNOWN"]
            for i in range(len(df)):
                flag = 0
                for s in series:
                    if s in df.iloc[i, 1]:
                        df.iloc[i, 1] = s
                        flag = 1
                        break
                    
                if flag == 0:
                    df.iloc[i, 1] = "UNKNOWN"

            df["產品系列"] = df["產品系列"].replace("其他", "other").replace("UNKNOWN", "Unknown")

            # 各產品每週總計
            df["week"] = pd.DatetimeIndex(df["預估交期"]).to_period("W").to_timestamp()
            df_g = df.groupby(["week", "產品系列"])["數量"].sum()
            df_g = df_g.unstack()
            df_g = df_g.sort_index()

            # 合併歷史及新資料
            flag = 1
            for idx in df_g.index:
                if idx <= df0.index[-1]: # 舊資料，取代
                    for col in df_g.columns:
                        if not pd.isna(df_g.loc[idx, col]):
                            df0.loc[idx, col] = df_g.loc[idx, col]
                else: # 新資料，合併
                    df_g = pd.concat([df0, df_g.loc[idx:]])
                    flag = 0
                    break

            if flag:
                df_g = df0.copy()
                

            # 填補缺失時間
            pdates = pd.date_range(start = df_g.index[0], end = df_g.index[-1], freq = "W-MON")
            df_g = df_g.reindex(pdates)
            df_g = df_g.reset_index()
            df_g =df_g.rename(columns = {"index": "week"})

            # 數量單位整數
            df_g.iloc[:, 1:] = df_g.iloc[:, 1:].astype(int)

            #save
            df_g.to_csv(self.origin_path, index = False)
            df_g.to_csv(self.output_path, index = False)

        except:
            logging.error(format_exc())
            df0.to_csv(self.output_path, index = False)

        finally:
            log.shutdown()



if __name__ == '__main__':

    if len(sys.argv) > 1: 
        input_ = sys.argv[1]
        input_ = base64.b64decode(input_).decode('utf-8')

        input_ = json.loads(input_)
    else:
        print("Input parameter error.")


    model_path = input_["model_path"]
    log_path = input_["log_path"]

    
    parser = Parser(model_path, log_path)
    output_path = parser.main()