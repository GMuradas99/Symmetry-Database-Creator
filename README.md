# Symmetry-Detabase-Creator

Script to create a database of images in grayscale with local symmetries and their labels with the purpose of training deep learning models. 
The algorithm uses handwritten digits from the MNIST database (downloaded from [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download)) to create
both symmetric and asymmetric figures. To create the database simply run `createDatabase.py`. This will automatically create a directory in the root folder with the data and 
its labels with the following structure:

```bash
├── ...
├── symmetry_database          # Root folder
│   ├── images                 # Folder containing the images. 
│   │   ├── 0.png
│   │   ├── 1.png
│   │   ├── ...
│   │   └── n.png
│   └── labels.csv             # File containing the labels for the symmetries and their backgrounds. 
└── ...
```

The images are generated with random backgrounds generated using Random Noise (algorithms adapted from Lode Vandevenne's [tutorial](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download) 

![97](https://user-images.githubusercontent.com/123949377/223097257-bf4ab52f-91db-46b6-bc54-28831ace2771.png)
