import json


def compare_json(path1, path2, reverse=False):
    with open(path1, "r", encoding="utf-8") as f:
        train_data1 = json.load(f)
    with open(path2, "r", encoding="utf-8") as f:
        train_data2 = json.load(f)

    eval1 = train_data1["eval"]
    eval2 = train_data2["eval"]
    for key, value in eval1.items():
        base_status1 = value[0]["base_status"]
        base_status2 = eval2[key][0]["base_status"]
        if base_status1 != base_status2:
            if base_status1 == 'fail' and not reverse:
                print("-" * 30)
                print(key)
                print("fail")
                print(value[0]['solution'])
                print("correct")
                print(eval2[key][0]['solution'])
            elif base_status1 == 'pass' and reverse:
                print("-" * 30)
                print(key)
                print("correct")
                print(value[0]['solution'])
                print("fail")
                print(eval2[key][0]['solution'])


            # else:
            #     print(eval2[key][0]['solution'])


if __name__ == "__main__":
    base_path = './gemma2'
    path1 = base_path + "/eval_results_default.json"
    path2 = base_path + "/eval_results_llm.json"
    path3 = base_path + "/eval_results_self_prompt.json"

    # compare_json(path1, path2)
    compare_json(path1, path2, True)
