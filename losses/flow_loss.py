import mlx.core as mx

def ot_cfm_loss(model, token_ids: mx.array, z_target: mx.array, mask: mx.array = None):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM) Loss.
    
    Translates the discrete tokens into a continuous coordinate space (x_1),
    samples Gaussian noise (x_0), and forces the model to predict the straight-line
    velocity vector field holding x_0 together toward x_1.
    """
    B, L = token_ids.shape
    
    # 1. Ground Truth Destination (The physical coordinates of the sentence)
    # Shape: (B, L, d_model)
    x_1 = model.embed_text(token_ids)
    
    # 2. Primordial Chaos (Standard Normal Distribution)
    # Shape: (B, L, d_model)
    x_0 = mx.random.normal(shape=x_1.shape)
    
    # 3. Time Sampling
    # We sample a continuous time scalar t ~ U(0, 1) for each sequence in the batch
    # Shape: (B, 1)
    t = mx.random.uniform(shape=(B, 1))
    
    # 4. Interpolate the exact intermediate state
    # We broadcast t from (B, 1) to (B, L, d_model) implicitly
    t_expand = mx.expand_dims(t, 2)  # (B, 1, 1)
    x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
    
    # 5. Optimal Transport Straight-Line Velocity
    # The true vector connecting x_0 to x_1
    v_true = x_1 - x_0
    
    # 6. Model Prediction (Blindfolded ODE solver step)
    v_pred = model(x_t, t, z_target, mask=mask)
    
    # 7. Mean Squared Error (Velocity difference)
    # Shape: (B, L, d_model)
    mse = mx.square(v_pred - v_true)
    
    # Average across the feature dimension first: (B, L)
    mse = mse.mean(axis=-1)
    
    if mask is not None:
        # Mask out padding tokens
        mse = mse * mask
        num_valid = mx.maximum(mask.sum(), 1.0)
        return mse.sum() / num_valid
        
    return mse.mean()
