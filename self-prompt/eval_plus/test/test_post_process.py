import keyboard
import time

def main():
    # 打开文件
    with open('0.py', 'r+', encoding='utf-8') as f:
        content = f.read()
        print(content)
        content = post_process_glm([content])
        # content = kill_space(content)

        f.write(content)


def post_process_glm(outputs):
    """
    This post-process method is only suitable for glm3 model.
    """
    try:
        outputs = outputs[0].split('python')[1]
    except IndexError as e:
        print("Raise ", e, ", cannot post_process")
        try:
            outputs = outputs[0].split('```')[1]
        except IndexError as e:
            print("Raise ", e, ", cannot post_process")
            return outputs

    # outputs = kill_test_cases(outputs)
    outputs = outputs.split('```')[0]

    return [outputs]


def kill_test_cases(outputs):
    index = outputs.rfind('return')
    test_case_index = outputs.find('\n', index)
    if test_case_index == -1:
        return outputs
    else:
        index = test_case_index

    outputs = outputs[:index+1]
    return outputs



if __name__ == '__main__':
    main()