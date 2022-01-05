from transformer import *

tmp_model = make_model(10, 10, 2)
for name, param in tmp_model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)
    else:
        print('no gradient necessary', name, param.data.shape)
