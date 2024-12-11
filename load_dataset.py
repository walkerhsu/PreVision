from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

def custom_collate_fn(batch):
    ids = [item["id"] for item in batch]
    images = [item["image"] for item in batch]
    conversations = [item["conversations"] for item in batch]
    
    return {
        "ids": ids,
        "images": images,
        "conversations": conversations
    }

def custom_dataloader(split):
    dataset: IterableDataset = load_dataset("ntudlcv/dlcv_2024_final1", split=split, streaming=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    return dataloader


if __name__ == "__main__":  
    # test usage
    train_loader = custom_dataloader("train")