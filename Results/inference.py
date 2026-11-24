"""
It is a template for loading a trained model checkpoint from disk and running inference on a dataset 
split without retraining. It sets up the model configuration, switches to evaluation mode without gradients, 
logs sample predictions and class probabilities alongside ground-truth labels, and computes overall accuracy.
"""

import sys
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "Results" else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from Configures import data_args, model_args, train_args
# Adjust imports below to match your actual dataset & model constructors:
# from load_dataset import get_dataloader
# from models.GnnNets import GnnNets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Configure dataset, model, and prototype flags to match the checkpoint
data_args.dataset_name = "MUTAG"
model_args.model_name = "gcn"
model_args.enable_prot = True  # False for Original, True for ProtGNN

# 2. Instantiate model and load saved weights
# model = GnnNets(...).to(device)
ckpt_path = PROJECT_ROOT / "checkpoint" / data_args.dataset_name / f"{model_args.model_name}_best.pth"
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# 3. Run test evaluation & inspect individual graph predictions
correct = 0
total = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):  # replace test_loader with your test split loader
        batch = batch.to(device)
        out = model(batch)  # inspect what your GnnNets.forward returns (logits, distances, prototypes, etc.)

        logits = out[0] if isinstance(out, (tuple, list)) else out
        preds = logits.argmax(dim=-1)
        labels = batch.y.view(-1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

        for i in range(min(3, labels.numel())):
            probs = torch.softmax(logits[i], dim=-1).cpu().tolist()
            print(
                f"Batch {batch_idx} | Graph {i}: "
                f"true={labels[i].item()}, pred={preds[i].item()}, probs={[round(p, 4) for p in probs]}"
            )

print(f"Standalone Test Accuracy: {correct / max(total, 1):.4f} ({correct}/{total})")