from transformers import pipeline, set_seed

model_path = '/data/team/zongwx1/llm_models/gpt-2-large'

def text_gen():
    generator = pipeline('text-generation', model=model_path)
    set_seed(42)
    prompt = """Generate one supplementary text that can help trigger the outputs based on the inputs: 
    {case}
    The text needs to be general enough to solve other different inputs.
    Your response must not contain the inputs and outputs before.
    Generate directly, no other useless response.
    Here is an example:
    "
    You are an intelligent programming assistant to produce Python algorithmic solutions.
    "
    The text is:"""

    results = generator(prompt, max_length=256, num_return_sequences=2)

    for i, result in enumerate(results):
        print(f"Generated Text {i + 1}: {result['generated_text']}\n")


def classification():
    generator = pipeline('text-classification', model=model_path)
    set_seed(42)
    prompt = "a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films."
    prompt = "apparently reassembled from the cutting-room floor of any given daytime soap ."

    results = generator(prompt, max_length=256)

    label = results[0]['label'][-1]

    print(label)

    print(results)


if __name__ == "__main__":
    classification()
