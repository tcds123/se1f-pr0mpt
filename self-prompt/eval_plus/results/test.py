import json


def main():
    with open('eval_results_true.json', 'r') as f:
        data1 = json.load(f)
    with open('eval_results_false.json', 'r') as f:
        data2 = json.load(f)
    index = 0
    while index <= 163:
        dict_index = 'HumanEval/' + str(index)
        result1 = data1['eval'][dict_index][0]
        result2 = data2['eval'][dict_index][0]
        index += 1
        if result1['base_status'] == result2['base_status'] or result1['base_status'] == 'pass':
            continue
        else:
            print(dict_index)
            print("-" * 20)
            print("True: \n", result1['solution'])
            print("-" * 20)
            print("False: \n", result2['solution'])
            print("-" * 20)



if __name__ == '__main__':
    main()