import torch

# LIVE 2 param

#liqe
A1 = 100.0
B1 = 0.0
C1 = 2.077636606942305
S1 = 1.101934158051146

#musiq
A2 = 100.0
B2 = 0.0
C2 = 43.539140163575430
S2 = 16.786801106500807

#ms_liqe
A3 = 100.0
B3 = 0.0
C3 = 1.984470232985137
S3 = 0.939839104077352

#clipiqa
A4= 100.0
B4 = 0.0
C4 = 0.383287415616992
S4 = 0.223077718907985


def logistic_mapping(x, model):
    A = 0
    B = 0
    C = 0
    S = 0
    if model == 'liqe':
        A = A1
        B = B1
        C = C1
        S = S1
    elif model == 'musiq':
        A = A2
        B = B2
        C = C2
        S = S2
    elif model == 'ms_liqe':
        A = A3
        B = B3
        C = C3
        S = S3  
    elif model == 'clipiqa':
        A = A4
        B = B4
        C = C4
        S = S4    

    z = (x - C) / S
    return (A - B) / (1 + torch.exp(-z)) + B