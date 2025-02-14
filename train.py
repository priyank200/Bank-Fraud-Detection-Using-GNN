#%%
import argparse
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import mask_select  # for masking node splits
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from model import GAT  # Import the GAT model
from torchviz import make_dot
from model2 import EdgeEnhancedGAT
from model import GAT

if torch.cuda.is_available():
    device = torch.device('cuda')
PARTIAL_DATA_DIR = "Small_test_data"
graph_data_path = f"{PARTIAL_DATA_DIR}/new_graph_data_2.pt"

# Load the graph dataset from graph.pt
graph_data = torch.load(graph_data_path, map_location=torch.device('cuda'), weights_only=False)
print(graph_data)

# Split the dataset: 70% nodes for training and 30% for testing
num_nodes = graph_data.num_nodes
train_mask = torch.rand(num_nodes) < 0.8 # randomly choose 80% of nodes for training
test_mask = ~train_mask  # the rest 20% for testing

# Define the GAT model
in_channels = 6  # Number of node features
hidden_channels = 16
out_channels = 2  # Binary classification
heads = 3  # Number of attention heads

# Initialize the GAT model
model = GAT(in_channels, hidden_channels, out_channels, heads).to(device)
# model = EdgeEnhancedGAT(in_channels,hidden_channels,out_channels,heads).to(device)

print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train(class_weights):
    model.train()
    optimizer.zero_grad()
    # Forward pass only on the nodes selected for training
    out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    # loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loss = F.cross_entropy(out[train_mask], graph_data.y[train_mask],weight=class_weights)  # Use only training nodes
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test():
    model.eval()
    # Perform inference on the test nodes only
    out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    pred = out[test_mask].argmax(dim=-1)  # Predictions only for test nodes
    correct = int((pred == graph_data.y[test_mask]).sum())
    acc = correct / test_mask.sum().item()  # Accuracy on test nodes
    return pred.cpu().numpy(), acc


# Create a unique directory for each run
# Start the timer and get the formatted start time
time_ = time.time()
formatted_time = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime(time_))
run_dir = f"result-MTP-2/run_{formatted_time}"
print(run_dir)
os.makedirs(run_dir, exist_ok=True)
print("Folder created...")

# Training loop
best_val_acc = final_test_acc = 0
epochs = 100
class_weights = torch.tensor([0.05209999904036522,0.9599999785423279], device=device)
for epoch in range(1, epochs + 1):
    loss = train(class_weights)
    predictions, test_acc = test()
    print(f"Epoch: {epoch}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Calculate metrics after the last test

# True labels from test data
true_labels = graph_data.y[test_mask].cpu().numpy()  

# Calculate confusion matrix and other metrics
conf_matrix = confusion_matrix(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average= 'weighted')
f1 = f1_score(true_labels, predictions)

# Calculate False Positive Rate (FPR) for each class
fpr = conf_matrix[1][1]/(conf_matrix[1][0]+conf_matrix[1][1])

# Save the confusion matrix as a PNG file
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()

# Annotate each cell in the confusion matrix with the corresponding count
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, f"{conf_matrix[i, j]}", ha="center", va="center", color="black",fontsize=15)

plt.savefig(f"{run_dir}/confusion_matrix.png")
plt.close()

# Save the results, including FPR, in a JSON file
results = {
    "graph_data": graph_data_path,
    "class_weights": class_weights.cpu().tolist(),
    "model_parameters": {
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "out_channels": out_channels,
        "heads": heads
    },
    "metrics": {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "final_test_accuracy": test_acc,
        "False Positive Rate": fpr  # Add FPR for each class
    },
    "epochs": epochs,
}
torch.save(model.state_dict(), f"{run_dir}/model.pth")
with open(f"{run_dir}/results.json", "w") as f:
    json.dump(results, f, indent=4)

# Print final metrics
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"False Positive Rate (FPR) per class: {fpr}")


# for roc_curve
fpr, tpr, _ = roc_curve(true_labels, predictions)

# Compute AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')

# Labels and legend
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()



# %%
