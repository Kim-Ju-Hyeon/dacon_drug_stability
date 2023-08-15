import torch
import scipy.spatial



def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    x = torch.cat([x, 2 * r * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * r * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

    tree = scipy.spatial.cKDTree(x.detach().numpy())
    _, col = tree.query(
        y.detach().numpy(), k=max_num_neighbors, distance_upper_bound=r + 1e-8)
    col = [torch.from_numpy(c).to(torch.long) for c in col]
    row = [torch.full_like(c, i) for i, c in enumerate(col)]
    row, col = torch.cat(row, dim=0), torch.cat(col, dim=0)
    mask = col < int(tree.n)
    return torch.stack([row[mask], col[mask]], dim=0)



def radius_graph(x,
                r,
                batch=None,
                loop=False,
                max_num_neighbors=32,
                flow='source_to_target'):
    assert flow in ['source_to_target', 'target_to_source']
    row, col = radius(x, x, r, batch, batch, max_num_neighbors + 1)
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)