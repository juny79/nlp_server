# placeholder for src/inference.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BartForConditionalGeneration
from src.utils import load_config
from src.model import load_model
from src.preprocess import DialoguePreprocessor

class TestDataset(Dataset):
    def __init__(self, encoder_input, test_id, length):
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item

def main():
    cfg = load_config("./config/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer, _ = load_model(cfg)
    
    # Load trained model from checkpoint
    model = BartForConditionalGeneration.from_pretrained(cfg["general"]["save_dir"])
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    print("Model loaded from checkpoint")

    # Load test data
    df = pd.read_csv(cfg["general"]["data_path"] + "test.csv")
    test_id = df["fname"]
    print(f"Loaded {len(df)} test samples")

    # Preprocess
    pre = DialoguePreprocessor(cfg["tokenizer"]["bos_token"], cfg["tokenizer"]["eos_token"])
    encoder_input_test = df["dialogue"].tolist()
    
    # Tokenize
    test_tokenized = tokenizer(
        encoder_input_test,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=cfg["tokenizer"]["encoder_max_len"],
        return_token_type_ids=False
    )
    
    test_dataset = TestDataset(test_tokenized, test_id, len(encoder_input_test))
    dataloader = DataLoader(test_dataset, batch_size=cfg["inference"]["batch_size"])
    
    # Run inference
    summary = []
    text_ids = []
    print("Starting inference...")
    
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = model.generate(
                input_ids=item['input_ids'].to(device),
                no_repeat_ngram_size=cfg["inference"]["no_repeat_ngram_size"],
                early_stopping=cfg["inference"]["early_stopping"],
                max_length=cfg["inference"]["generate_max_length"],
                num_beams=cfg["inference"]["num_beams"],
            )
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)
    
    # Clean summaries
    remove_tokens = cfg["inference"]["remove_tokens"]
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token, " ") for sentence in preprocessed_summary]
    
    # Save results
    df_out = pd.DataFrame({
        "fname": df["fname"],
        "summary": preprocessed_summary
    })
    
    output_path = cfg["inference"].get("result_path", "./") + "output.csv"
    df_out.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
