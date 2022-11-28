import sys, base64, json


if len(sys.argv) > 2: 
    mode = sys.argv[1]
    input_ = sys.argv[2]
    input_ = base64.b64decode(input_).decode('utf-8')

    input_ = json.loads(input_)
else:
    print("Input parameter error.")



if mode == "train":
    model_path = input_["model_path"]
    log_path = input_["log_path"]


    from data_parser import Parser
    parser = Parser(model_path, log_path)
    parser.main()


    from train import Model
    model_ = Model(model_path, log_path)
    scores = model_.main()

elif mode == "predict":
    start = int(input_["START"])
    end = int(input_["END"])
    model_path = input_["MODEL"]
    output_path = input_["OUTPUT"]
    log_path = input_["LOG"]

    from predict import Predict
    predict = Predict(start, end, model_path, output_path, log_path)
    results = predict.main()

else:
    print('Input "MODE" does not exist.')