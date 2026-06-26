import argparse

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def generate(ckpt_path: str, prompt: str, max_new_tokens: int) -> str:
    config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        trust_remote_code=True,
        torch_dtype=config.torch_dtype,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs.to(model.device),
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path")
    parser.add_argument(
        "--prompt",
        default=(
            "An attention function can be described as mapping a query and a set "
            "of key-value pairs to an output, where the query, keys, values, and "
            "output are all vectors. The output is"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    print(generate(args.ckpt_path, args.prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()
