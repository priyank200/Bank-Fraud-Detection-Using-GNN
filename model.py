# %%
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torchviz import make_dot


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6, edge_dim=3)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6,edge_dim=3)

    def forward(self, x, edge_index, edge_feat):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index,edge_feat))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index,edge_feat)
        return x


# %%
# model = GAT(in_channels=7,hidden_channels=16,out_channels=2,heads=3)
# print(model)

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# PARTIAL_DATA_DIR = "Small_test_data"
# graph_data_path = f"{PARTIAL_DATA_DIR}/graph_data.pt"

# # Load the graph dataset from graph.pt
# graph_data = torch.load(graph_data_path).to(device)
# print(graph_data)

# # Define the GAT model
# in_channels = 7  # Number of node features
# hidden_channels = 16
# out_channels = 2  # Binary classification
# heads = 3  # Number of attention heads

# # Initialize the GAT model
# model = GAT(in_channels, hidden_channels, out_channels, heads).to(device)
# print(model)

#%% Example input tensor to visualize the model structure
# sample_input = graph_data.x  # Using graph data input features as a sample input
# sample_output = model(sample_input, graph_data.edge_index, graph_data.edge_attr)
# make_dot(sample_output, params=dict(model.named_parameters())).render("model_architecture", format="png")
# # %%

# # Prepare the input and output names
# input_names = ['input']  # You can choose a more descriptive name
# output_names = ['output']  # You can also choose a more descriptive name

# # Export the model to ONNX format
# torch.onnx.export(
#     model,
#     (sample_input, graph_data.edge_index, graph_data.edge_attr),  # Tuple of inputs
#     'model.onnx',  # Specify the filename
#     input_names=input_names,
#     output_names=output_names,
#     export_params=True,  # Store the trained parameter weights inside the model file
#     opset_version=16,  # Specify the ONNX version
# )


# # %%
# import onnx

# # Load the ONNX model
# model = onnx.load("model.onnx")

# # Print the model's graph
# print(onnx.helper.printable_graph(model.graph))

# # %%
