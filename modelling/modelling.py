from model.randomforest import RandomForest
from model.chained_multi_output import *



def model_predict(data, df, name):
    results = []

    # First Model - Random Forest
    print("RandomForest")
    model_rf = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model_rf.train(data)
    model_rf.predict(data.X_test)
    model_rf.print_results(data)

    # Second Model - Chained-Multi output with Random Forest base
    label_columns = ["y2", "y3", "y4"]
    run_second_model(data, df, label_columns)


def model_evaluate(model, data):
    model.print_results(data)