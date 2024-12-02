import json



if __name__ == "__main__":
    target = """['?\\n'], [':\\n'], [' True'], [' otherwise'], [' that'], ['1'], [' takes'], [' \"\"\"\\n\\n'], [' '], ['   '], ['cba'], ['('], [' \"\"\"'], ['_length'], ['``'], ['\\n'], [' =='], ['```'], ['0'], [' i'], ['ens'], [' return'], ['Can'], [' if'], ['user'], [' is'], [' returns'], ['orange'], [' %'], ['\\n\\n'], [' range'], [' ='], [' number'], [' following'], ['Hello'], ['k'], [' prime'], ["('"], ['assistant'], [' for'], ['       '], [' or'], [' you'], ['2'], [' length'], ['abcd'], [' Python'], ['itt'], ['<|endoftext|>'], [' l']"""
    target = """['Hello'], [''], ['i'], ['<|user|>'], ['number'], ['\\n'], ['2'], ['function'], ['ange'], ['           '], ['if'], ['ab'], ['prime'], ['cd'], [''], ['for'], ['len'], ['Ex'], ['def'], ['am'], [','], ['('], ['itt'], ['otherwise'], ['that'], [':'], ['l'], ['if'], ["')"], ['='], ['Write'], ['or'], ['%'], ['   '], ['return'], ['a'], ['range'], ['=='], ['k'], ['\"\"\"'], ['ens'], ['ba'], ['ples'], ['False'], ['returns'], ['in'], ['_'], ['0'], ['1'], ['True'], """
    target_list = eval(target)
    target_list = [i for i in target_list]
    for list in target_list:
        list = [list[0].replace('\n', '\\n')]
        # print(f'\'{list[0]}\'')

        print(list[0], end=', ')
