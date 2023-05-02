import torch

from model.modeling_t5.py import attention_mask_encoder_new

id_of_type = torch.Tensor([1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0]).reshape(1, -1)
id_of_row = torch.Tensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]).reshape(1, -1)

gm_mask = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]).reshape(1, -1)

id_of_col = torch.Tensor([0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0]).reshape(1, -1)
edges = ((1, 1), (2, 2), (1, 2), (2, 1), (3, 1), (3, 2), (1, 3), (2, 3))

interested_positions = torch.BoolTensor([[
    [1,1,1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,0,0,1,0,0,0],
    [1,1,1,1,1,0,1,0,0,1,0,0],
    [1,1,1,1,1,0,0,1,0,0,1,0],
    [1,1,1,0,0,1,1,1,1,0,0,0],
    [1,1,0,1,0,1,1,1,0,1,0,0],
    [1,1,0,0,1,1,1,1,0,0,1,0],
    [1,1,1,0,0,1,0,0,1,1,1,0],
    [1,1,0,1,0,0,1,0,1,1,1,0],
    [1,1,0,0,1,0,0,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0]
]])

am = attention_mask_encoder_new(gm_mask, id_of_type, id_of_row, id_of_col, edges)
