import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Cargando el conjunto de datos
train = datasets.MNIST('', train=True, download=True, 
                        transform=transforms.Compose([
                            transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=True, 
                        transform=transforms.Compose([
                            transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 64) # 28*28 es el digito de nuestras imagenes y 64 son las conexiones 
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 10) # Toma 64 conexiones pero produce solo 10 por las 10 clases que tenemos en la data

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return F.log_softmax(x, dim=1)

net = Net()
print(net)

loss_function = nn.CrossEntropyLoss()   # ¿Qué tan lejos esta del valor deseado?
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Entrenamiento

for epoch in range(3):                  # 3 recorridas por toda la data
    for data in trainset:               # data es el lote de la data
        X, y = data                     # X es el lote de entrada, y es el lote de valores desados
        net.zero_grad()                 # Conjunto de gradientes en 0 antes del calculo de perdidas
        output = net(X.view(-1, 784))   # Pasa en el lote reformado
        loss = F.nll_loss(output, y)    # almacena en loss el valor de perdida
        loss.backward()                 # Ajusta los parametros de acuerdo a la perdida
        optimizer.step()                # Optimiza los pesos
    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1, 784))
        #print(output)
        for idx, i in enumerate(output):
            #print(torch.argmax(i), y[idx])
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Exactitud: ", round(correct/total, 3))

plt.imshow(X[0].view(28,28))
plt.show()

print(torch.argmax(net(X[0].view(-1,784))[0]))