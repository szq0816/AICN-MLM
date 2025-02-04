## Ambiguous Instance-Aware Contrastive Network with Multi-Level Matching for Multi-View Document Clustering



Requirements:
To run this project, ensure the following dependencies are installed:

    python==3.7.11
    pytorch==1.9.0
    numpy==1.20.1
    scikit-learn==0.22.2.post1
    scipy==1.6.2

To test the trained model, run:
```bash
python test.py
```
You will get results in the following format:

KUST-CB：ACC = 0.7512 NMI = 0.8422 ARI = 0.6863 PUR=0.8203
KUST-CT：ACC = 0.7700 NMI = 0.8467 ARI = 0.7097 PUR=0.8391
KUST-BT：ACC = 0.8369 NMI = 0.8734 ARI = 0.7897 PUR=0.8514
KUST-CV：ACC = 0.7485 NMI = 0.8332 ARI = 0.6868 PUR=0.8176
KUST-ET：ACC = 0.8921 NMI = 0.8634 ARI = 0.8022 PUR=0.8921
KUST-CL：ACC = 0.7419 NMI = 0.8234 ARI = 0.6711 PUR=0.8110
KUST-CE：ACC = 0.8726 NMI = 0.8692 ARI = 0.7901 PUR=0.8726
Reuters：ACC = 0.6206 NMI = 0.4259 ARI = 0.3372 PUR=0.6307

To train a new model, run:
```bash
python train.py
```

Note: Due to variations in hardware and random initialization, 
the performance of the model may differ when run on different devices or with different random seeds.
Note that the optimal parameters on our local setup might not be directly applicable to other environments.

# Dataset and pretrained model
To obtain the dataset and pre-trained model of this project, please use the following link:

https://pan.baidu.com/s/1RNkNm-q9yKZ6KuBxX9JOQw

Extraction code: 8ris.
And place the dataset in the data directory and the pre-trained model in the models directory.

# Contact
If you have any questions, please feel free to contact me at: 1911235620@qq.com
