# %%
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# %% Define Data Folder 
DATA_DIR = "SAML"
PARTIAL_DATA_DIR = "Small_test_data"


# %%
class GraphDataset:
    """
        - Class to create a graph dataset for the transactions trails
    """
    def __init__(self, csv_file):
        """
            - csv_file: Give the path of the .csv file
        """
        self.path = csv_file
        self.data = self._read_csv_file()
        self.account_to_index = {} # For mapping account to a idx
        self.edge_index = None # To store the edge index example ([[1,2][3,4]]) -> 1-3, 2-4
        self.features = None # To store the features of the node
        self.labels = None # To store the target labels
        self.edge_features = None # To store the features of the edge
        self._create_account_mapping()
        self._create_node_features_and_labels()
        self._create_edges()
    
    def _read_csv_file(self):
        print("Reading the Data ...")
        dtype = {
            'Sender_account': 'int64',
            'Receiver_account': 'int64',
            'Sender_bank_location': 'int8',
            'Receiver_bank_location': 'int8',
            'Payment_currency': 'int8',
            'Received_currency': 'int8',
            'Amount': 'float32',
            'Payment_type': 'int8',
            'Year': 'int16',
            'Month': 'int8',
            'Day': 'int8',
            'Is_laundering': 'int8',
            'Laundering_type':'int8'
        }
        return pd.read_csv(self.path, dtype=dtype)


    def _create_account_mapping(self):
        """Create a mapping from account to node index."""
        print("Creating a Map...")
        unique_accounts = pd.concat([self.data['Sender_account'], self.data['Receiver_account']]).unique()
        self.account_to_index = {account: idx for idx, account in enumerate(unique_accounts)}
        # print(len(self.account_to_index))

    def _create_edges(self):
        """Generate edges and edge features based on transactions."""
        edges = []
        edge_features = []

        for _, row in tqdm(self.data.iterrows(), total=self.data.shape[0], desc='Processing edges'):
            sender_idx = self.account_to_index[row['Sender_account']]
            receiver_idx = self.account_to_index[row['Receiver_account']]
            edges.append([sender_idx, receiver_idx])

            # Edge features: Amount, Payment Type, and Date
            edge_features.append(self._create_edge_features(row))

        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.edge_features = torch.tensor(edge_features, dtype=torch.float)

    def _create_edge_features(self, row):
        """Extract edge features from the row data."""
        return [
            row['Amount'],
            row['Payment_type'],  # Optionally, encode this value as categorical if necessary
            row['Year'] + row['Month'] / 12 + row['Day'] / 365,  # Simple numeric date representation
        ]

    def _create_node_features_and_labels(self):
        """Generate features and labels for each node based on account transactions."""
        num_accounts = len(self.account_to_index)
        self.features = torch.zeros((num_accounts, 6))  # 7 features per node
        self.labels = torch.zeros(num_accounts, dtype=torch.long)

        for account, idx in tqdm(self.account_to_index.items(), total=num_accounts, desc='Processing nodes and labels'):
            # Extract node features
            self.features[idx], laundering_label = self._calculate_node_features_and_label(account)

            # Set label
            self.labels[idx] = torch.tensor(laundering_label, dtype=torch.long)

    def _calculate_node_features_and_label(self, account):
        """Calculate node features and the laundering label for the given account."""
        # Transactions where the account is the sender or receiver
        sender_txns = self.data[self.data['Sender_account'] == account]
        receiver_txns = self.data[self.data['Receiver_account'] == account]

        # Default values for node features
        sender_location = 18 # Here 18 represent No location
        receiver_location = 18 # Here 18 represent No location
        payment_currency = 12 # Here if there is no sender location means no payment curracny so 12
        received_currency = 12 # Here if there is no receive location means no payment curracny so 12
        outdegree = 0 # Number of time send money
        indegree = 0 # Number of time receive money
        # suspicious_count = 0 # How many launddering trancation this account is involved

        # Feature 1: Sender bank location and outdegree
        if not sender_txns.empty:
            sender_location = sender_txns['Sender_bank_location'].mode()[0]
            outdegree = sender_txns.shape[0]
            payment_currency = sender_txns['Payment_currency'].mode()[0]

        # Feature 2: Receiver bank location and indegree
        if not receiver_txns.empty:
            receiver_location = receiver_txns['Receiver_bank_location'].mode()[0]
            indegree = receiver_txns.shape[0]
            received_currency = receiver_txns['Received_currency'].mode()[0]

        # # Feature 7: Suspicious transaction count
        # suspicious_count = self.data[
        #     (self.data['Sender_account'] == account) | (self.data['Receiver_account'] == account)
        # ]['Is_laundering'].sum()

        # Label: 1 if the account is involved in any laundering transaction, else 0
        laundering_label = 1 if self.data[
            (self.data['Sender_account'] == account) | (self.data['Receiver_account'] == account)
        ]['Is_laundering'].sum() > 0 else 0

        # Handle isolated nodes
        if outdegree == 0 and indegree == 0:
            return torch.tensor([0, 0, 0, 0, 0, 0, 0]), laundering_label

        # Return the feature vector and laundering label
        return torch.tensor([
            sender_location,
            receiver_location,
            payment_currency,
            received_currency,
            outdegree,
            indegree,
        ]), laundering_label

    def get_graph_data(self):
        """Return the graph data."""
        return Data(x=self.features, edge_index=self.edge_index, edge_attr=self.edge_features, y=self.labels)
    
    def save_graph_data(self, graph_data, save_path):
        """Save the graph data as a .pt file."""
        # graph_data = self.get_graph_data()
        torch.save(graph_data, save_path)
        print(f"Graph data saved to {save_path}")


dataset_paths = [f"{PARTIAL_DATA_DIR}/New_data_2.csv"]
save_paths = [f"{PARTIAL_DATA_DIR}/new_graph_data_2.pt"]

# %% Start Create Graph dataset and save them
for dataset, save_path in zip(dataset_paths, save_paths):
    graph_dataset = GraphDataset(dataset)
    graph_data = graph_dataset.get_graph_data()
    print(graph_data)
    
    # Save the graph data correctly
    graph_dataset.save_graph_data(graph_data=graph_data, save_path=save_path)
    
    del graph_data  # Clear graph_data to free up memory

# %%
