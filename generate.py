import torch
from transformers import GPT2TokenizerFast

from src.config import GPTConfig
from src.model import MiniGPT


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=1.0
):
    model.eval()

    device = next(model.parameters()).device

    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    )

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep only the model's context window
            input_ids_cond = input_ids[:, -model.config.max_seq_len:]

            logits = model(input_ids_cond)

            # Get logits for the last token
            logits = logits[:, -1, :]

            # Temperature
            logits = logits / temperature

            # Convert logits into probabilities
            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            # Sample next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Append token
            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

    return tokenizer.decode(
        input_ids[0].tolist()
    )


def main():

    config = GPTConfig()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = MiniGPT(config)

    checkpoint = torch.load(
        "checkpoints/best.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    tokenizer = GPT2TokenizerFast.from_pretrained(
        "gpt2"
    )

    prompt = "ROMEO:"

    text = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=200,
        temperature=0.8
    )

    print("\nGenerated text:\n")
    print(text)


if __name__ == "__main__":
    main()