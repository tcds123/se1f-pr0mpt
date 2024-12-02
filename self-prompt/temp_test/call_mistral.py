def mistral_call():
    from mistral_inference.model import Transformer
    from mistral_inference.generate import generate

    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest


    tokenizer = MistralTokenizer.from_file("/data/team/zongwx1/llm_models/mistral-7B-Instruct-v0.3/tokenizer.model.v3")  # change to extracted tokenizer file
    model = Transformer.from_folder("/data/team/zongwx1/llm_models/mistral-7B-Instruct-v0.3")  # change to extracted model dir

    completion_request = ChatCompletionRequest(messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")])

    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    out_tokens, _ = generate([tokens], model, max_tokens=512, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

    print(result)


def hf_call():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"  # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained("/data/team/zongwx1/llm_models/mistral-7B-Instruct-v0.3")
    tokenizer = AutoTokenizer.from_pretrained("/data/team/zongwx1/llm_models/mistral-7B-Instruct-v0.3")

    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant",
         "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])


if __name__ == "__main__":
    mistral_call()