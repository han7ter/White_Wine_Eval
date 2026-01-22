import torch

X_train = torch.load("X_train.pt")
X_val = torch.load("X_val.pt")
y_train = torch.load("y_train.pt")
y_val = torch.load("y_val.pt")
W = torch.randn(11, 1, requires_grad = True)
b = torch.randn(1, 1, requires_grad = True)
alpha = 0.025

for _ in range(X_train.shape[0]):
    y_hat = X_train @ W + b
    loss = ((y_hat - y_train) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        W -= alpha * W.grad
        b -= alpha * b.grad
    W.grad = None
    b.grad = None



with torch.no_grad():
    y_hat = X_val @ W + b
score = (y_hat - y_val).abs().mean()
print(score)

# альфа коэфициенты результаты:
# 0.001 = 0.5964
# 0.005 = 0.5798
# 0.01 = 0.5790
# 0.025 = 0.5783
# ИТОГО: альфа 0.025 победила юхууууууууу
