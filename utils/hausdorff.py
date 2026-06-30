import torch

def hausdorff_distance_gpu(pred, target, distance="euclidean"):
    """
    GPU Hausdorff distance between two binary segmentation masks.

    Args:
        pred (torch.Tensor): predicted binary mask, shape (B, H, W).
        target (torch.Tensor): ground-truth binary mask, shape (B, H, W).
        distance (str): 'euclidean' or 'chessboard'. Default 'euclidean'.

    Returns:
        hd (float): Hausdorff distance value.
    """

    assert pred.shape == target.shape, "pred and target must have the same shape"

    # collect positive pixel coords (value=1) from pred and target
    pred_points = torch.nonzero(pred).float()  # shape: (num_points, 2)
    target_points = torch.nonzero(target).float()  # shape: (num_points, 2)

    if pred_points.size(0) == 0 or target_points.size(0) == 0:
        return torch.tensor(0.0, device=pred.device)  # no positive pixels -> 0

    # pairwise distance matrix
    pred_expanded = pred_points.unsqueeze(1)  # (num_pred_points, 1, 2)
    target_expanded = target_points.unsqueeze(0)  # (1, num_target_points, 2)

    # compute distance from each pred point to each target point
    if distance == "euclidean":
        dist_matrix = torch.sqrt(torch.sum((pred_expanded - target_expanded) ** 2, dim=2))  # (num_pred_points, num_target_points)
    elif distance == "chessboard":
        dist_matrix = torch.max(torch.abs(pred_expanded - target_expanded), dim=2)[0]  # (num_pred_points, num_target_points)
    else:
        raise ValueError("Invalid distance type. Use 'euclidean' or 'chessboard'.")

    # Hausdorff: max over min in each direction
    hd_pred_to_target = torch.max(torch.min(dist_matrix, dim=1)[0])  # max of min distance pred -> target
    hd_target_to_pred = torch.max(torch.min(dist_matrix, dim=0)[0])  # max of min distance target -> pred

    hd = torch.max(hd_pred_to_target, hd_target_to_pred)
    return hd