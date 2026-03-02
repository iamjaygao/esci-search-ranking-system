import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# ==========================================
# 1. Feature Extraction Pipeline
# ==========================================
def extract_esci_features(examples_path, products_path):
    print("Loading raw ESCI Parquet files...")
    df_ex = pd.read_parquet(examples_path)
    df_pr = pd.read_parquet(products_path)
    
    # Merge on product_id and locale
    print("Merging datasets...")
    df = pd.merge(df_ex, df_pr, how='inner', on=['product_id', 'product_locale'])
    
    # Filter strictly for the training split to avoid leaking test data
    df = df[df['split'] == 'train'].copy()
    
    print(f"Extracting features for {len(df)} training rows. This may take a minute...")
    
    # --- A. Relevance Features ---
    # (In a real pipeline, you merge your actual BM25/Vector scores here. 
    # We simulate them for the training setup so the code runs out-of-the-box)
    df['bm25_score'] = np.random.uniform(0, 1, len(df))
    df['semantic_score'] = np.random.uniform(-1, 1, len(df))
    
    # Exact word overlap ratio (intersection of words / query length)
    def calc_overlap(row):
        q_words = set(str(row['query']).lower().split())
        t_words = set(str(row['product_title']).lower().split())
        if len(q_words) == 0: return 0.0
        return len(q_words.intersection(t_words)) / len(q_words)
    
    df['word_overlap'] = df.apply(calc_overlap, axis=1)
    
    # --- B. Length / Vagueness Features ---
    df['query_length'] = df['query'].astype(str).apply(lambda x: len(x.split()))
    df['title_length'] = df['product_title'].astype(str).apply(lambda x: len(x.split()))
    
    # --- C. Quality Features ---
    df['has_brand'] = df['product_brand'].notna().astype(float)
    df['bullet_count'] = df['product_bullet_point'].astype(str).apply(lambda x: len(x.split('\n')) if x != 'None' else 0)
    
    # --- D. Implicit Popularity Features ---
    # 1. Product Frequency (How many queries trigger this product?)
    prod_counts = df.groupby('product_id')['query_id'].transform('count')
    df['log_product_freq'] = np.log1p(prod_counts)
    
    # 2. Brand Frequency (How dominant is this brand in the dataset?)
    brand_counts = df.groupby('product_brand')['product_id'].transform('count')
    df['log_brand_freq'] = np.log1p(brand_counts.fillna(0))
    
    # --- Target Labels ---
    # Map ESCI letters to regression targets
    label_map = {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0}
    df['target_score'] = df['esci_label'].map(label_map).fillna(0.0)
    
    # Drop rows with NaNs in our specific feature columns to be safe
    feature_cols = [
        'bm25_score', 'semantic_score', 'word_overlap', 
        'query_length', 'title_length', 'has_brand', 
        'bullet_count', 'log_product_freq', 'log_brand_freq'
    ]
    df = df.dropna(subset=feature_cols)
    
    return df, feature_cols

# ==========================================
# 2. The Deep Neural Network
# ==========================================
class DeepESCIReranker(nn.Module):
    def __init__(self, input_dim):
        super(DeepESCIReranker, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1: Expand to 128 dimensions to catch complex feature crossings
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), # Normalizes outputs to stabilize training
            nn.ReLU(),
            nn.Dropout(0.2),     # Drops 20% of neurons randomly to prevent overfitting
            
            # Layer 2: Compress to 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 3: Compress to 32
            nn.Linear(64, 32),
            nn.ReLU(),
            
            # Layer 4: Output Score
            nn.Linear(32, 1),
            nn.Sigmoid()         # Binds the final output between 0.0 and 1.0
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 3. Dataset & Training Loop
# ==========================================
class ESCIDataset(Dataset):
    def __init__(self, df, feature_cols):
        # Normalize features (Crucial for deep networks with large numbers like word counts)
        features = df[feature_cols].values
        self.features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        self.labels = df['target_score'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

def train_model():
    examples_file = "esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
    products_file = "esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet"
    
    if not os.path.exists(examples_file):
        print(f"Error: Could not find {examples_file}. Check your directory structure.")
        return
        
    df_train, feature_columns = extract_esci_features(examples_file, products_file)
    input_size = len(feature_columns)
    
    print(f"\nInitializing DataLoader with {input_size} features...")
    dataset = ESCIDataset(df_train, feature_columns)
    
    # Use a large batch size (1024) to train faster on the massive dataset
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    model = DeepESCIReranker(input_dim=input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf') # Start with an infinitely high loss
    epochs = 5
    print("\nStarting Deep Training Loop...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "esci_reranker.pth")
            print("--> Network improved! Saving new best weights.")
        
    print("Training complete. Model is ready for evaluation.")
    # Load the absolute best weights back into RAM before returning
    model.load_state_dict(torch.load("esci_reranker.pth"))
    return model

if __name__ == "__main__":
    trained_model = train_model()