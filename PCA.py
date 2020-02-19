import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('USArrests.csv')
data = data.dropna()
Nombres = np.array(data['Unnamed: 0'])
data = data.drop("Unnamed: 0",axis=1)
arrows = ['Murder', 'Assault', 'UrbanPop', 'Rape']

dataT = np.array(np.transpose(data))
#Normalizacion
for i in range(len(dataT[:,0])):
    dataT[i,:] = (dataT[i,:] - np.mean(dataT[i,:]))/np.std(dataT[i,:])

cov = np.cov(dataT)

eigenvalues, eigenvectors = np.linalg.eig(cov)

eigenvalues = np.sort(eigenvalues)[::-1]


losmios1 = eigenvectors[:,0]
losmios2 = eigenvectors[:,1]

elX = np.dot(losmios1,dataT)
elY = np.dot(losmios2,dataT)
plt.figure(0,figsize=(15,15))
for i in range(len(elX)):
    plt.text(elX[i],elY[i],Nombres[i])
for i in range(len(eigenvalues)):
    plt.arrow(0,0,eigenvectors[i,0],eigenvectors[i,1],head_width=0.01, head_length=.01)
    plt.text(eigenvectors[i,0],eigenvectors[i,1],arrows[i],fontsize=12)

plt.xlabel('First PC')
plt.xlim(min(elX),max(elX))
plt.ylim(min(elY),max(elY))
plt.ylabel('Second PC')
plt.savefig('arrestos.png')
plt.close()

#Grafica de la varianza
plt.figure(1,figsize=(15,15))
losPorcentajes = eigenvalues/np.sum(eigenvalues)
print(np.sum(losPorcentajes))
for i in range(len(eigenvalues)-1):
    losPorcentajes[i+1] = losPorcentajes[i+1]+losPorcentajes[i]
losx = np.array(range(4))
losx[:] = losx[:]+1
plt.plot(losx,losPorcentajes*100)
plt.xlabel('EigenValue Number')
plt.ylabel('Porcentaje de la Varianza')
plt.grid(1)
plt.savefig('varianza_arrestos.png')
plt.close()


#Cars
data = pd.read_csv('Cars93.csv')
data = data.dropna()
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
dataC = np.array(data[columns])
modelo = np.array(data['Model'])
dataCT = np.array(np.transpose(dataC))
#Normalizacion
for i in range(len(dataCT[:,0])):
    dataCT[i,:] = (dataCT[i,:] - np.mean(dataCT[i,:]))/np.std(dataCT[i,:])

covC = np.cov(dataCT)

eigenvaluesC, eigenvectorsC = np.linalg.eig(covC)
eigenvaluesC = np.sort(eigenvaluesC)[::-1]
losmios1C = eigenvectorsC[:,0]
losmios2C= eigenvectorsC[:,1]

elXC = np.dot(losmios1C,dataCT)
elYC = np.dot(losmios2C,dataCT)

plt.figure(0, figsize=(15,15))
for i in range(len(elXC)):
    plt.text(elXC[i],elYC[i],modelo[i])
for i in range(len(eigenvaluesC)):
    plt.arrow(0,0,eigenvectorsC[i,0],eigenvectorsC[i,1], head_width=0.01, head_length=0.01 )
    plt.text(eigenvectorsC[i,0],eigenvectorsC[i,1],columns[i],fontsize=12)

plt.xlabel('First PC')
plt.xlim(min(elX),max(elX))
plt.ylim(min(elY),max(elY))
plt.ylabel('Second PC')
plt.savefig('cars.png')
plt.close()


plt.figure(3,figsize=(15,15))
losPorcentajesC = eigenvaluesC/np.sum(eigenvaluesC)
for i in range(len(eigenvaluesC)-1):
    losPorcentajesC[i+1] = losPorcentajesC[i+1]+losPorcentajesC[i]
losxC = np.array(range(len(eigenvaluesC)))
losxC[:] = losxC[:]+1
plt.plot(losxC,losPorcentajesC*100)
plt.xlabel('EigenValue Number')
plt.ylabel('Porcentaje de la Varianza')
plt.grid(1)
plt.savefig('varianza_cars.png')
plt.close()