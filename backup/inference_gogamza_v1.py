from tqdm import tqdm

def generate_summary(dialog):
    dialog = clean_dialogue(dialog)
    prompt = f"<SYS> {dialog} <SUMMARY>"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_length=120,
        min_length=30,
        num_beams=6,
        no_repeat_ngram_size=3,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        length_penalty=1.1,
        early_stopping=True
    )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    summary = summary.replace("<pad>", "").strip()
    return summary

# test dataset load
test_df = pd.read_csv("./data/test.csv")

preds = []
for dialog in tqdm(test_df["dialogue"]):
    preds.append(generate_summary(dialog))

test_df["summary"] = preds
test_df.to_csv("submission.csv", index=False)
