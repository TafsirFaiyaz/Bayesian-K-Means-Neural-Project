import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # Fallback for UMAP
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available; using PCA fallback.")
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Differentiable K-Means Layer (with Gumbel noise for uncertainty)
class DifferentiableKMeans(nn.Module):
    def __init__(self, n_clusters, feature_dim, temperature=5.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.temperature = temperature
        # Smaller random init for stability
        self.centers = nn.Parameter(torch.randn(n_clusters, feature_dim) * 0.1)
        # Orthogonal with low gain
        if feature_dim >= n_clusters:
            nn.init.orthogonal_(self.centers, gain=0.1)
        # Higher noise for diversity
        self.centers.data += torch.randn_like(self.centers) * 0.05
        
    def forward(self, x):
        distances = torch.cdist(x.unsqueeze(1), self.centers.unsqueeze(0)).squeeze(1)
        logits = -distances / self.temperature
        if self.training:
            # Gumbel noise for more stochasticity
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            logits += gumbel_noise
        assignments = F.softmax(logits, dim=1)
        return assignments, distances

# BDKM-NF Model (smaller hidden_dim=64, added contrastive head)
class BDKMNF(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=64, num_clusters=10, temperature=5.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # Reduced layers for simplicity
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim)
        )
        
        self.kmeans_layer = DifferentiableKMeans(num_clusters, hidden_dim, temperature)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
        # Contrastive head: projector for NT-Xent
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        self.prior_mu = torch.zeros(num_clusters, hidden_dim)
        self.prior_std = torch.ones(num_clusters, hidden_dim)
        
    def reparameterize(self, z, noise_std=0.5):
        if self.training:
            eps = torch.randn_like(z) * noise_std
            return z + eps
        return z
    
    def forward(self, x, noise_std=0.5):
        h = self.encoder(x.view(-1, self.input_dim))
        z = self.reparameterize(h, noise_std)
        assignments, distances = self.kmeans_layer(z)
        x_hat = self.decoder(z)
        kl_centers = self.compute_center_kl()
        proj = self.projector(z)  # For contrastive
        return x_hat, assignments, distances, z, kl_centers, proj
    
    def compute_center_kl(self):
        centers = self.kmeans_layer.centers
        prior_dist = torch.distributions.Normal(self.prior_mu.to(centers.device), 
                                              self.prior_std.to(centers.device))
        post_dist = torch.distributions.Normal(centers, torch.ones_like(centers) * 0.05)
        kl = torch.distributions.kl_divergence(post_dist, prior_dist).sum()
        return kl

# Baseline: Simple Bayesian Clustering
class BayesianClustering(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=64, num_clusters=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        self.cluster_layer = nn.Linear(hidden_dim, num_clusters)
        
    def forward(self, x):
        h = self.encoder(x.view(-1, 784))
        cluster_logits = self.cluster_layer(h)
        assignments = F.softmax(cluster_logits, dim=1)
        x_hat = self.decoder(h)
        return x_hat, assignments, h

# Loss function (BCE recon + contrastive + silhouette-aware, FIXED INDEXING)
def compute_loss(model, x, beta=0.5, tau=0.07):
    x_hat, assignments, distances, z, kl_centers, proj = model(x)
    # Binary cross-entropy for better scaling on [0,1] pixels
    recon_loss = F.binary_cross_entropy(x_hat, x.view(-1, 784))
    # Encourage confident assignments (lower entropy)
    assignment_entropy = -torch.sum(assignments * torch.log(assignments + 1e-10), dim=1).mean()
    # Stronger balance
    cluster_sizes = assignments.sum(dim=0)
    probs = cluster_sizes / cluster_sizes.sum()
    balance_loss = -torch.sum(probs * torch.log(probs + 1e-10))
    # Silhouette approximation (intra vs inter) - FIXED: Use 1D boolean indexing
    batch_size = z.size(0)
    pred_labels = torch.argmax(assignments, dim=1)
    intra_dists = torch.zeros(batch_size, device=z.device)
    inter_dists = torch.zeros(batch_size, device=z.device)
    for i in range(batch_size):
        same_cluster = (pred_labels == pred_labels[i])  # 1D boolean
        if same_cluster.sum() > 1:  # Need at least 2 for intra dist
            same_z = z[same_cluster]
            intra_dists[i] = torch.cdist(same_z, same_z)[0, 1:].mean()  # Exclude self
        else:
            intra_dists[i] = 0.0  # Default low intra if singleton
        
        inter_mask = (pred_labels != pred_labels[i])  # 1D boolean
        if inter_mask.sum() > 0:
            inter_z = z[inter_mask]
            inter_dists[i] = torch.cdist(inter_z, z[i:i+1]).min(dim=0)[0].mean()  # Min dist to nearest inter
        else:
            inter_dists[i] = 10.0  # Default high inter if no others
    
    a = intra_dists.mean()  # Avg intra
    b = inter_dists.mean()  # Avg inter
    sil_approx = (b - a).clamp(min=0) / (a + b + 1e-8)  # Softmax-like clamp
    sil_loss = -sil_approx  # Minimize negative silhouette
    # Contrastive (NT-Xent simplified: assume augmented view as positive)
    pos_sim = F.cosine_similarity(proj, proj.roll(1, dims=0), dim=-1).mean()  # Shifted as pseudo-positive
    neg_sim = (F.cosine_similarity(proj.unsqueeze(1), proj.unsqueeze(0), dim=-1) - torch.eye(batch_size, device=proj.device)).mean()
    contrastive_loss = -torch.log(torch.exp(pos_sim / tau) / (torch.exp(pos_sim / tau) + torch.exp(neg_sim / tau)))
    # Adjusted weights
    total_loss = recon_loss + beta * kl_centers + 0.1 * assignment_entropy + 0.5 * balance_loss + 0.2 * sil_loss + 0.5 * contrastive_loss
    return total_loss, recon_loss, kl_centers, assignment_entropy

# Training function (temperature annealing, more epochs)
def train_model(model, train_loader, val_loader, epochs=25, lr=5e-4, beta=0.5):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Temperature annealing
        current_temp = 5.0 * (0.95 ** epoch)
        model.kmeans_layer.temperature = current_temp
        
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            loss, _, _, _ = compute_loss(model, data, beta)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
            # Debug logging (reduced frequency)
            if batch_idx % 200 == 0:
                with torch.no_grad():
                    sample_assign = model(data)[1]
                    sample_pred = torch.argmax(sample_assign, dim=1)
                    print(f"Epoch {epoch+1}, Batch {batch_idx}: Unique clusters = {len(torch.unique(sample_pred))}, Temp: {current_temp:.2f}")
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                loss, _, _, _ = compute_loss(model, data, beta)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return train_losses, val_losses

# Evaluation functions (with UMAP preprocessing + post-processing K-means)
def evaluate_clustering(model, data_loader, true_labels=None):
    model.eval()
    all_embeddings = []
    all_assignments = []
    all_true_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            _, assignments, _, z, _, _ = model(data)
            all_embeddings.append(z.numpy())
            all_assignments.append(assignments.numpy())
            if true_labels is not None:
                all_true_labels.append(labels.numpy())
    
    embeddings = np.vstack(all_embeddings)
    assignments = np.vstack(all_assignments)
    pred_labels_soft = np.argmax(assignments, axis=1)
    
    # UMAP/PCA preprocessing for better clustering
    if UMAP_AVAILABLE:
        reducer = UMAP(n_components=10, random_state=42)
    else:
        reducer = PCA(n_components=10, random_state=42)
    embeddings_reduced = reducer.fit_transform(embeddings)
    
    # Post-processing: Hard K-means on reduced embeddings
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10).fit(embeddings_reduced)
    pred_labels = kmeans.labels_
    
    unique_labels = len(np.unique(pred_labels))
    print(f"Number of unique predicted labels (post-KMeans): {unique_labels}")
    
    if unique_labels > 1:
        sil_score = silhouette_score(embeddings_reduced, pred_labels)
    else:
        print("Warning: Only 1 unique cluster detected. Setting silhouette score to -1.0.")
        sil_score = -1.0
    
    results = {'silhouette': sil_score}
    
    if true_labels is not None:
        true_labels = np.concatenate(all_true_labels)
        ari_score = adjusted_rand_score(true_labels, pred_labels)
        nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
        results.update({'ari': ari_score, 'nmi': nmi_score})
    
    return results, embeddings, assignments, pred_labels

# Uncertainty analysis (higher noise)
def analyze_uncertainty(model, data_loader, n_samples=10):
    model.train()
    all_uncertainties = []
    
    for data, _ in data_loader:
        uncertainties = []
        for _ in range(n_samples):
            with torch.no_grad():
                _, assignments, _, _, _, _ = model(data, noise_std=0.5)
                uncertainties.append(assignments.numpy())
        
        uncertainties = np.stack(uncertainties)
        uncertainty = np.var(uncertainties, axis=0).mean(axis=1)
        all_uncertainties.extend(uncertainty)
    
    return np.array(all_uncertainties)

# Visualization functions
def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curves')
    plt.show()

def plot_clusters_tsne(embeddings, pred_labels, true_labels=None, uncertainties=None):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, axes = plt.subplots(1, 2 if true_labels is not None else 1, figsize=(15, 6))
    if true_labels is None:
        axes = [axes]
    
    s = 50 if uncertainties is None else 50 / (uncertainties[:len(embeddings_2d)] + 1e-8)
    scatter = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=pred_labels, cmap='tab10', s=s)
    axes[0].set_title('Predicted Clusters (t-SNE)')
    plt.colorbar(scatter, ax=axes[0])
    
    if true_labels is not None:
        scatter2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                  c=true_labels, cmap='tab10', s=50)
        axes[1].set_title('True Labels (t-SNE)')
        plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

def main():
    set_seed(42)
    
    # Enhanced transform with data augmentation for training
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),
        transforms.RandomRotation(10),  # ±10 deg
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5)  # Slight shear/translate
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,))
    ])
    
    # Load full train dataset WITH augmentations (so split subsets inherit them implicitly)
    full_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                            download=True, transform=train_transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                            download=True, transform=test_transform)
    
    # Split: Both train/val will get augs (acceptable for simplicity)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # DataLoaders: No 'transform' arg needed (inherited from datasets)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    print("Dataset loaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    bdkm_model = BDKMNF(input_dim=784, hidden_dim=64, num_clusters=10, temperature=5.0)
    baseline_model = BayesianClustering(input_dim=784, hidden_dim=64, num_clusters=10)
    
    print("\nTraining BDKM-NF model...")
    train_losses, val_losses = train_model(bdkm_model, train_loader, val_loader, 
                                         epochs=25, lr=5e-4, beta=0.5)
    
    bdkm_model.load_state_dict(torch.load('best_model.pth'))
    
    plot_training_curves(train_losses, val_losses)
    
    print("\nEvaluating BDKM-NF model...")
    results, embeddings, assignments, pred_labels = evaluate_clustering(
        bdkm_model, test_loader, true_labels=True)
    
    print(f"BDKM-NF Results:")
    print(f"Silhouette Score: {results['silhouette']:.4f}")
    print(f"ARI Score: {results['ari']:.4f}")
    print(f"NMI Score: {results['nmi']:.4f}")
    
    print("\nAnalyzing uncertainty...")
    uncertainties = analyze_uncertainty(bdkm_model, test_loader, n_samples=10)
    print(f"Average uncertainty: {np.mean(uncertainties):.4f}")
    
    true_labels = []
    for _, labels in test_loader:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)
    
    print("\nGenerating visualizations...")
    plot_clusters_tsne(embeddings, pred_labels, true_labels, uncertainties)
    
    print("\nTraining baseline model...")
    def train_baseline(model, train_loader, val_loader, epochs=25, lr=5e-4):
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        for epoch in range(epochs):
            model.train()
            for data, _ in train_loader:
                optimizer.zero_grad()
                x_hat, assignments, h = model(data)
                recon_loss = F.binary_cross_entropy(x_hat, data.view(-1, 784))
                entropy_loss = -torch.sum(assignments * torch.log(assignments + 1e-10), dim=1).mean()
                loss = recon_loss + 0.5 * entropy_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Baseline Epoch {epoch+1}/{epochs}')
    
    train_baseline(baseline_model, train_loader, val_loader)
    
    def evaluate_baseline(model, data_loader):
        model.eval()
        all_embeddings = []
        all_pred_labels = []
        all_true_labels = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                _, assignments, h = model(data)
                pred_labels_batch = np.argmax(assignments.numpy(), axis=1)
                all_embeddings.append(h.numpy())
                all_pred_labels.extend(pred_labels_batch)
                all_true_labels.extend(labels.numpy())
        
        embeddings = np.vstack(all_embeddings)
        pred_labels = np.array(all_pred_labels)
        true_labels = np.array(all_true_labels)
        
        # UMAP/PCA + post-processing K-means
        if UMAP_AVAILABLE:
            reducer = UMAP(n_components=10, random_state=42)
        else:
            reducer = PCA(n_components=10, random_state=42)
        embeddings_reduced = reducer.fit_transform(embeddings)
        
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10).fit(embeddings_reduced)
        pred_labels = kmeans.labels_
        
        unique_labels = len(np.unique(pred_labels))
        print(f"Baseline number of unique predicted labels (post-KMeans): {unique_labels}")
        
        if unique_labels > 1:
            sil_score = silhouette_score(embeddings_reduced, pred_labels)
        else:
            print("Warning: Only 1 unique cluster detected in baseline. Setting silhouette score to -1.0.")
            sil_score = -1.0
        
        ari_score = adjusted_rand_score(true_labels, pred_labels)
        nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
        
        return {'silhouette': sil_score, 'ari': ari_score, 'nmi': nmi_score}
    
    baseline_results = evaluate_baseline(baseline_model, test_loader)
    
    print(f"\nBaseline Results:")
    print(f"Silhouette Score: {baseline_results['silhouette']:.4f}")
    print(f"ARI Score: {baseline_results['ari']:.4f}")
    print(f"NMI Score: {baseline_results['nmi']:.4f}")
    
    comparison_df = pd.DataFrame({
        'Model': ['BDKM-NF', 'Baseline'],
        'Silhouette': [results['silhouette'], baseline_results['silhouette']],
        'ARI': [results['ari'], baseline_results['ari']],
        'NMI': [results['nmi'], baseline_results['nmi']]
    })
    
    print("\nComparison Results:")
    print(comparison_df.to_string(index=False))
    
    print(f"\nBDKM-NF shows improvement over baseline:")
    print(f"Silhouette: {results['silhouette'] - baseline_results['silhouette']:.4f}")
    print(f"ARI: {results['ari'] - baseline_results['ari']:.4f}")
    print(f"NMI: {results['nmi'] - baseline_results['nmi']:.4f}")
    
    return {
        'bdkm_results': results,
        'baseline_results': baseline_results,
        'embeddings': embeddings,
        'uncertainties': uncertainties,
        'comparison_df': comparison_df
    }

if __name__ == "__main__":
    results = main()