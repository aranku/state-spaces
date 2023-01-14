from src.dataloaders.synthetic import Copying

print("ok")

data = Copying().dataset_train
for k in data:
  print(k)
  break