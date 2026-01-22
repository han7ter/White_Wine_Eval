import torch

data = torch.load("wine-white.pt")

index = torch.randperm(data.shape[0])
data = data[index]

X, y = data[:, :-1], data[:, -1:]
X_train, X_val = X[:3000], X[3000:]
y_train, y_val = y[:3000], y[3000:]

mean = X_train.mean(0)
std = X_train.std(0)
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

torch.save(X_train, "X_train.pt")
torch.save(y_train, "y_train.pt")
torch.save(X_val, "X_val.pt")
torch.save(y_val, "y_val.pt")
