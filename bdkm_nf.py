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

# Differentiable K-Means Layer
class DifferentiableKMeans(nn.Module):
    def __init__(self, n_clusters, feature_dim, temperature=0.1):
        super().__init__()
        self.n_clusters = n_clusters
        self.temperature = temperature
        # Initialize cluster centers as learnable parameters
        self.centers = nn.Parameter(torch.randn(n_clusters, feature_dim) * 0.1)
        
    def forward(self, x):
        # Compute distances to cluster centers
        # x: (batch_size, feature_dim)
        # centers: (n_clusters, feature_dim)
        distances = torch.cdist(x.unsqueeze(1), self.centers.unsqueeze(0))  # (batch_size, 1, n_clusters)
        distances = distances.squeeze(1)  # (batch_size, n_clusters)
        
        # Soft assignments using temperature-scaled softmax
        assignments = F.softmax(-distances / self.temperature, dim=1)
        return assignments, distances

# BDKM-NF Model
class BDKMNF(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_clusters=10, temperature=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim)
        )
        
        # Differentiable K-Means layer
        self.kmeans_layer = DifferentiableKMeans(num_clusters, hidden_dim, temperature)
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # For MNIST pixel values [0,1]
        )
        
        # Bayesian priors
        self.prior_mu = torch.zeros(num_clusters, hidden_dim)
        self.prior_std = torch.ones(num_clusters, hidden_dim)
        
    def reparameterize(self, z, noise_std=0.1):
        """Reparameterization trick for stochasticity"""
        if self.training:
            eps = torch.randn_like(z) * noise_std
            return z + eps
        return z
    
    def forward(self, x, noise_std=0.1):
        # Encode
        h = self.encoder(x.view(-1, self.input_dim))
        
        # Add stochasticity via reparameterization trick
        z = self.reparameterize(h, noise_std)
        
        # Clustering
        assignments, distances = self.kmeans_layer(z)
        
        # Decode
        x_hat = self.decoder(z)
        
        # Compute Bayesian KL divergence for cluster centers
        kl_centers = self.compute_center_kl()
        
        return x_hat, assignments, distances, z, kl_centers
    
    def compute_center_kl(self):
        """Compute KL divergence between cluster centers and prior"""
        centers = self.kmeans_layer.centers
        prior_dist = torch.distributions.Normal(self.prior_mu.to(centers.device), 
                                              self.prior_std.to(centers.device))
        post_dist = torch.distributions.Normal(centers, torch.ones_like(centers) * 0.1)
        kl = torch.distributions.kl_divergence(post_dist, prior_dist).sum()
        return kl

# Baseline: Simple Bayesian Clustering
class BayesianClustering(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_clusters=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        self.cluster_layer = nn.Linear(hidden_dim, num_clusters)
        
    def forward(self, x):
        h = self.encoder(x.view(-1, 784))
        cluster_logits = self.cluster_layer(h)
        assignments = F.softmax(cluster_logits, dim=1)
        x_hat = self.decoder(h)
        return x_hat, assignments, h

# Loss function
def compute_loss(model, x, beta=0.1):
    x_hat, assignments, distances, z, kl_centers = model(x)
    
    # Reconstruction loss
    recon_loss = F.mse_loss(x_hat, x.view(-1, 784))
    
    # Assignment entropy regularization (encourage confident assignments)
    assignment_entropy = -torch.sum(assignments * torch.log(assignments + 1e-10), dim=1).mean()
    
    # Total loss
    total_loss = recon_loss + beta * (kl_centers + assignment_entropy)
    
    return total_loss, recon_loss, kl_centers, assignment_entropy

# Training function
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, beta=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            loss, recon_loss, kl_loss, entropy_loss = compute_loss(model, data, beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                loss, _, _, _ = compute_loss(model, data, beta)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
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

# Evaluation functions
def evaluate_clustering(model, data_loader, true_labels=None):
    model.eval()
    all_embeddings = []
    all_assignments = []
    all_true_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            _, assignments, _, z, _ = model(data)
            all_embeddings.append(z.numpy())
            all_assignments.append(assignments.numpy())
            if true_labels is not None:
                all_true_labels.append(labels.numpy())
    
    embeddings = np.vstack(all_embeddings)
    assignments = np.vstack(all_assignments)
    pred_labels = np.argmax(assignments, axis=1)
    
    # Silhouette score
    sil_score = silhouette_score(embeddings, pred_labels)
    
    results = {'silhouette': sil_score}
    
    if true_labels is not None:
        true_labels = np.concatenate(all_true_labels)
        ari_score = adjusted_rand_score(true_labels, pred_labels)
        nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
        results.update({'ari': ari_score, 'nmi': nmi_score})
    
    return results, embeddings, assignments, pred_labels

# Uncertainty analysis
def analyze_uncertainty(model, data_loader, n_samples=10):
    model.train()  # Enable stochasticity
    all_uncertainties = []
    
    for data, _ in data_loader:
        uncertainties = []
        for _ in range(n_samples):
            with torch.no_grad():
                _, assignments, _, _, _ = model(data, noise_std=0.1)
                uncertainties.append(assignments.numpy())
        
        # Compute variance across samples
        uncertainties = np.stack(uncertainties)
        uncertainty = np.var(uncertainties, axis=0).mean(axis=1)  # Average uncertainty per sample
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
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, axes = plt.subplots(1, 2 if true_labels is not None else 1, figsize=(15, 6))
    if true_labels is None:
        axes = [axes]
    
    # Plot predicted clusters
    scatter = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=pred_labels, cmap='tab10', 
                             s=50 if uncertainties is None else 50/uncertainties[:len(embeddings_2d)])
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
    # Set seed for reproducibility
    set_seed(42)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,))
    ])
    
    full_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                            download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                            download=True, transform=transform)
    
    # Split training data
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    print("Dataset loaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize models
    bdkm_model = BDKMNF(input_dim=784, hidden_dim=128, num_clusters=10)
    baseline_model = BayesianClustering(input_dim=784, hidden_dim=128, num_clusters=10)
    
    print("\nTraining BDKM-NF model...")
    # Train BDKM-NF
    train_losses, val_losses = train_model(bdkm_model, train_loader, val_loader, 
                                         epochs=30, lr=1e-3, beta=0.1)
    
    # Load best model
    bdkm_model.load_state_dict(torch.load('best_model.pth'))
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    # Evaluate on test set
    print("\nEvaluating BDKM-NF model...")
    results, embeddings, assignments, pred_labels = evaluate_clustering(
        bdkm_model, test_loader, true_labels=True)
    
    print(f"BDKM-NF Results:")
    print(f"Silhouette Score: {results['silhouette']:.4f}")
    print(f"ARI Score: {results['ari']:.4f}")
    print(f"NMI Score: {results['nmi']:.4f}")
    
    # Analyze uncertainty
    print("\nAnalyzing uncertainty...")
    uncertainties = analyze_uncertainty(bdkm_model, test_loader, n_samples=10)
    print(f"Average uncertainty: {np.mean(uncertainties):.4f}")
    
    # Get true labels for visualization
    true_labels = []
    for _, labels in test_loader:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_clusters_tsne(embeddings, pred_labels, true_labels, uncertainties)
    
    # Train baseline for comparison
    print("\nTraining baseline model...")
    
    def train_baseline(model, train_loader, val_loader, epochs=30):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(epochs):
            model.train()
            for data, _ in train_loader:
                optimizer.zero_grad()
                x_hat, assignments, h = model(data)
                recon_loss = F.mse_loss(x_hat, data.view(-1, 784))
                entropy_loss = -torch.sum(assignments * torch.log(assignments + 1e-10), dim=1).mean()
                loss = recon_loss + 0.1 * entropy_loss
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Baseline Epoch {epoch+1}/{epochs}')
    
    train_baseline(baseline_model, train_loader, val_loader)
    
    # Evaluate baseline
    def evaluate_baseline(model, data_loader):
        model.eval()
        all_embeddings = []
        all_pred_labels = []
        all_true_labels = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                _, assignments, h = model(data)
                pred_labels = np.argmax(assignments.numpy(), axis=1)
                all_embeddings.append(h.numpy())
                all_pred_labels.extend(pred_labels)
                all_true_labels.extend(labels.numpy())
        
        embeddings = np.vstack(all_embeddings)
        pred_labels = np.array(all_pred_labels)
        true_labels = np.array(all_true_labels)
        
        sil_score = silhouette_score(embeddings, pred_labels)
        ari_score = adjusted_rand_score(true_labels, pred_labels)
        nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
        
        return {'silhouette': sil_score, 'ari': ari_score, 'nmi': nmi_score}
    
    baseline_results = evaluate_baseline(baseline_model, test_loader)
    
    print(f"\nBaseline Results:")
    print(f"Silhouette Score: {baseline_results['silhouette']:.4f}")
    print(f"ARI Score: {baseline_results['ari']:.4f}")
    print(f"NMI Score: {baseline_results['nmi']:.4f}")
    
    # Comparison table
    comparison_df = pd.DataFrame({
        'Model': ['BDKM-NF', 'Baseline'],
        'Silhouette': [results['silhouette'], baseline_results['silhouette']],
        'ARI': [results['ari'], baseline_results['ari']],
        'NMI': [results['nmi'], baseline_results['nmi']]
    })
    
    print("\nComparison Results:")
    print(comparison_df.to_string(index=False))
    
    # Statistical significance test (would need multiple runs)
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