import torch

# def hyperbolic_dist(emb1,emb2):
#     diff = emb1 - emb2
#     euclidean_dist = torch.norm(diff)
    
#     u_norm2 = torch.norm(emb1)**2
#     v_norm2 = torch.norm(emb2)**2
    
#     #product to get (1 - ||u||^2)*(1 - ||v||^2)
#     denom = (1 - u_norm2) * (1 - v_norm2) + 1e-5  # add epsilon to avoid division by zero
        
#     # Hyperbolic distance (Poincaré ball metric):
#     # d(u,v) = arccosh( 1 + 2*||u-v||^2/((1-||u||^2)(1-||v||^2)) )
#     x = 1 + 2 * (euclidean_dist**2) / denom
#     # Clamp to ensure the argument of acosh is >= 1
#     x = x.clamp(min=1 + 1e-5)
#     hyperbolic_dist = torch.acosh(x)
    
#     return hyperbolic_dist


# def seq_euclidean_dist(eseq1,eseq2,single_dist):
#     vector = torch.zeros(len(eseq1)) 
#     for item in range(len(eseq1)):
#         vector[item]=single_dist(eseq1[item],eseq2[item])
#     eseq_dist = torch.norm(vector)

#     return eseq_dist

# def pairseq_dist_affinity(seq,single_dist):
#     n_users = len(seq)
#     pariseq_dist = torch.zeros((n_users,n_users))
#     for user1 in range(n_users):
#         for user2 in range(user1,n_users):
#             pariseq_dist[user1,user2] = seq_euclidean_dist(seq[user1],seq[user2],single_dist)
#     pariseq_dist += pariseq_dist.T

#     affinity = torch.exp(-pariseq_dist**2)

#     return affinity




# def laplacian_sum_torch(f_x, dist):
#     d = dist.sum(axis=1)  
#     sqrt_d = torch.sqrt(d)
#     f_x_normalized = f_x / sqrt_d  
    
#     diff = f_x_normalized[:, None] - f_x_normalized[None, :]
    
#     weighted_sq_diff = dist * (diff ** 2)
    
#     quad_form = 0.5 * torch.sum(weighted_sq_diff)
#     return quad_form

           
def hyperbolic_dist(emb1, emb2):
    """
    Compute hyperbolic distance in the Poincaré ball model for batched inputs.
    emb1: (batch_size, num_items, emb_dim)
    emb2: (batch_size, num_items, emb_dim)
    """
    diff = emb1 - emb2  # (batch_size, num_items, emb_dim)
    euclidean_dist_sq = torch.sum(diff ** 2, dim=-1)  # (batch_size, num_items)
    
    u_norm_sq = torch.sum(emb1 ** 2, dim=-1)  # (batch_size, num_items)
    v_norm_sq = torch.sum(emb2 ** 2, dim=-1)  # (batch_size, num_items)

    denom = (1 - u_norm_sq) * (1 - v_norm_sq) + 1e-5  # (batch_size, num_items)
    x = 1 + 2 * euclidean_dist_sq / denom  # (batch_size, num_items)
    x = torch.clamp(x, min=1 + 1e-5)  # Ensure valid input for acosh
    hyperbolic_dist = torch.acosh(x)  # (batch_size, num_items)

    return hyperbolic_dist

def seq_euclidean_dist(eseq1, eseq2, single_dist):
    """
    Compute sequence-wise Euclidean distance using a distance function (batched).
    eseq1, eseq2: (batch_size, num_items, emb_dim)
    """
    dist_vector = single_dist(eseq1, eseq2)  # (batch_size, num_items)
    eseq_dist = torch.norm(dist_vector, dim=-1)  # (batch_size)
    return eseq_dist

def pairseq_dist_affinity(seq, single_dist):
    """
    Compute the pairwise sequence distance affinity matrix efficiently.
    seq: (batch_size, num_items, emb_dim)
    single_dist: function to compute distance between two sequences
    """
    batch_size = seq.shape[0]
    
    # Expand dimensions to compute pairwise distances efficiently
    seq1 = seq.unsqueeze(1)  # (batch_size, 1, num_items, emb_dim)
    seq2 = seq.unsqueeze(0)  # (1, batch_size, num_items, emb_dim)

    pairwise_distances = seq_euclidean_dist(seq1, seq2, single_dist)  # (batch_size, batch_size)
    
    affinity = torch.exp(-pairwise_distances ** 2)  # (batch_size, batch_size)
    
    return affinity





def hyperbolic_distance_matrix(embeddings):
    """
    Compute pairwise hyperbolic distances for item embeddings in the Poincaré ball.
    embeddings: Tensor of shape (N, emb_dim)
    Returns: Tensor of shape (N, N) containing pairwise hyperbolic distances.
    """
    # Compute squared Euclidean distances between embeddings
    diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # (N, N, emb_dim)
    euclidean_dist_sq = torch.sum(diff ** 2, dim=2)  # (N, N)
    
    # Compute norms for each embedding
    E_norm_sq = torch.sum(embeddings ** 2, dim=1, keepdim=True)  # (N, 1)
    # Denominator for each pair (1 - ||u||^2)*(1 - ||v||^2)
    denom = (1 - E_norm_sq) * (1 - E_norm_sq).t() + 1e-5  # (N, N)
    
    # Compute the argument of arccosh
    x = 1 + 2 * euclidean_dist_sq / denom
    x = torch.clamp(x, min=1 + 1e-5)  # ensure valid input for arccosh
    dist = torch.acosh(x)  # (N, N)
    return dist

def euclidean_distance_matrix(embeddings):
    """
    Compute pairwise Euclidean distances for item embeddings.
    embeddings: Tensor of shape (N, emb_dim)
    Returns: Tensor of shape (N, N) containing pairwise Euclidean distances.
    """
    diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # (N, N, emb_dim)
    dist = torch.norm(diff, dim=2)  # (N, N)
    return dist

def manifold_regularization_embeddings(embeddings, metric='hyperbolic', sigma=1.0, lambda_reg=1.0):
    """
    Regularizes the embeddings directly using a Laplacian-like term.
    
    embeddings: Tensor of shape (N, emb_dim)
    metric: 'euclidean' or 'hyperbolic'
    sigma: parameter for the Gaussian affinity function
    lambda_reg: scaling factor for the regularization term
    
    Returns: A scalar regularization loss.
    """
    if metric == 'euclidean':
        dist = euclidean_distance_matrix(embeddings)
    elif metric == 'hyperbolic':
        dist = hyperbolic_distance_matrix(embeddings)
    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'hyperbolic'.")

    # Build affinity matrix: A_{ij} = exp(-dist(i,j)^2 / sigma^2)
    A = torch.exp(-dist ** 2 / sigma ** 2)
    
    # Laplacian regularization term: sum_{i,j} A_{ij} * ||E_i - E_j||^2
    diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # (N, N, emb_dim)
    sq_diff = torch.sum(diff ** 2, dim=2)  # (N, N)
    
    reg_loss = (A * sq_diff).sum()
    return lambda_reg * reg_loss
