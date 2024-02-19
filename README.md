# Real-time Domain Adaptation in Semantic Segmentation

This repository contains the code of our project for the course "Advanced Machine Learning" at Politecnico di Torino. The goal of the project is to become familiar with the domain adaptation problem in the context of semantic segmentation, by implementing and testing different strategies.

#### AUTHORS
- s317320 - Buccoliero Elena
- s309234 - Chiabodo Alessandro
- s314971 - Occhiena Beatrice

## Project structure
- `datasets/`: classes to handle the datasets used in the project.
  - [Cityscapes](datasets/cityscapes.py)
  - [GTA5](datasets/gta5.py)

- `model/`: implementation of the models used in the project and their components.
  - [BiSeNet](model/model_stages.py)
    - AttentionRefinement module
    - ContextPath module
    - FeatureFusion module
  - [BiSeNet Discriminator](model/model_stages.py)
  - [STDCNet813 Backbone](model/stdcnet.py)

- `training/`: code to implement the different training strategies used in the project.
  - [Simple Training](training/simple_train.py) - Trains the model on the source domain and tests it on the target domain.
  - [Single Layer Domain Adaptation](training/single_layer_da_train.py) - Performs unsupervised adversarial training with labeled synthetic data and unlabelled real-world data.
  - [Fourier Domain Adaptation](training/fda_train.py) - Performs semi-supervised training using the enhanced synthetic data generated by FDA.
  - [Self-Learning](training/fda_self_learning_train.py) - Performs self-learning using pseudo-labels generated by the model adapted with MBT.

- `utils/`: utility functions and classes used in the various stages of the project.
  - [General Functions](utils/general.py) - Various utility functions, Save and Load Checkpoints
  - [Data Augmentation](utils/aug.py) - Various Transformations
  - [FDA](utils/fda.py) - Style Transfer functions, Entropy Minimization Loss
    - MBT Adaptation
    - Pseudo-Label Generation

- `eval.py`: code to evaluate the performance of the models on the target domain.
- `main.py`: code to initialize the training/testing process according to the chosen arguments.
- `test_style_transfer.ipynb`: code to perform a quick test of the FDA style transfer and the strong data aumentation.
- `Project Overview.pdf`: the compiled version of the original project requirements file.

## Requirements
- The two datasets are not included in the repository, but they can be downloaded from [datasets_drive_link](https://drive.google.com/drive/u/0/folders/1iE8wJT7tuDOVjEBZ7A3tOPZmNdroqG1m).
- The pre-trained weights for STDCNet813 are available at [stdcnet_drive_link](https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1).

## Steps and Results
1. **RELATED WORKS** - Read the papers and understand the methods.
2. **TESTING REAL-TIME SEMANTIC SEGMENTATION**
    - > A - Define the upper bound for the domain adaptation phase.
    
      Train the model on the the training set of Cityscapes and evaluate it on the validation set of Cityscapes.

      ```bash
      main.py --dataset Cityscapes --data_transformations 0 --batch_size 20 --learning_rate 0.005 --num_epochs 50 --save_model_path trained_models\cityscapes --mode train --num_workers 4 --optimizer sgd
      ```

      | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |----------------|------------|-----------------------------|
      | 80.0           | 53.4       | 1:15                        |

    - > B - Train on the synthetic dataset.
    
      Train the model on the training set of GTA5 and evaluate it on the validation set of GTA5.

      ```bash
      main.py --dataset GTA5 --data_transformations 0 --batch_size 20 --learning_rate 0.005 --num_epochs 50 --save_model_path trained_models\gta --mode train --num_workers 4 --optimizer sgd
      ```

      | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |----------------|------------|-----------------------------|
      | 75.8           | 47.8       | 1:35                        |

    - > C1 - Evaluate the domain shift.

      Test the model trained at step B on the validation set of Cityscapes.

      ```bash
      main.py --dataset CROSS_DOMAIN --mode test --data_transformations 0 --resume_model_path trained_models\gta\best.pth 
      ```

      | Accuracy _(%)_ | mIoU _(%)_ |
      |----------------|------------|
      | 44.8           | 17.9       |

    - > C2 - Evaluate the domain shift performing data augmentation.

      Train the model on the training set of GTA5 using data augmentation (probability 0.5) and evaluate it on the validation set of Cityscapes.

      ```bash
      main.py --mode train --dataset CROSS_DOMAIN --data_transformations 1 --batch_size 20 --learning_rate 0.008 --num_epochs 50 --save_model_path trained_models\augm1 --num_workers 4 --optimizer sgd
      ```
      ```bash
      main.py --dataset CROSS_DOMAIN --data_transformations 2 --batch_size 20 --learning_rate 0.008 --num_epochs 50 --save_model_path trained_models\augm2 --mode train --num_workers 4 --optimizer sgd
      ```

      | Augmentation        | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |---------------------|----------------|------------|-----------------------------|
      | Weak augmentation   | 41.1           | 20.7       | 1:35                        |
      | Strong augmentation | 70.3           | 29.7       | 1:40                        |


3. **IMPLEMENTING UNSUPERVISED ADVERSARIAL DOMAIN ADAPTATION** - Perform adversarial training with labeled synthetic data (GTA5) and unlabelled real-world data (Cityscapes). Evaluate the model on the validation set of Cityscapes.

    ```bash
    main.py --mode train_da --dataset CROSS_DOMAIN --data_transformations 0 --batch_size 10 --learning_rate 0.005 --num_epochs 50 --save_model_path trained_models\norm_da --num_workers 4 --optimizer sgd --d_lr 0.0001
    ```
    ```bash
    main.py --mode train_da --dataset CROSS_DOMAIN --data_transformations 1 --batch_size 10 --learning_rate 0.005 --num_epochs 50 --save_model_path trained_models\norm_da --num_workers 4 --optimizer sgd --d_lr 0.0001
    ```
    ```bash
    main.py --mode train_da --dataset CROSS_DOMAIN --data_transformations 2 --batch_size 10 --learning_rate 0.005 --num_epochs 50 --save_model_path trained_models\norm_da --num_workers 4 --optimizer sgd --d_lr 0.0001
    ```

    | Augmentation        | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
    |---------------------|----------------|------------|-----------------------------|
    | No augmentation     | 65.0           | 27.2       | 2:18                        |
    | Weak augmentation   | 68.9           | 28.4       | 2:32                        |
    | Strong augmentation | 72.1           | 30.9       | 2:40                        |


4. **IMPROVEMENTS - option c**
    - > A - Implement a fast image-to-image translation algorithm like FDA.

      Apply FDA to enhance the GTA5 training images, i.e. swap the low-frequency components of the Fourier amplitude spectra of the source and target images. The parameter *beta* controls the size of the low frequency window to be replaced.

      ![Alt text](images/image-1.png)

      Train the model with labeled enhanced synthetic data from GTA5 and unlabelled real-world data from Cityscapes (semi-supervised learning). Evaluate the model on the validation set of Cityscapes.

      ```bash
      main.py  --mode train_fda --dataset CROSS_DOMAIN --batch_size 10 --learning_rate 0.01 --num_epochs 50 --save_model_path trained_models\norm_fda0.09 --num_workers 4 --optimizer sgd --beta 0.09
      ```
      ```bash
      main.py  --mode train_fda --dataset CROSS_DOMAIN --batch_size 10 --learning_rate 0.01 --num_epochs 50 --save_model_path trained_models\norm_fda0.05 --num_workers 4 --optimizer sgd --beta 0.05
      ```
      ```bash
      main.py  --mode train_fda --dataset CROSS_DOMAIN --batch_size 10 --learning_rate 0.01 --num_epochs 50 --save_model_path trained_models\norm_fda0.01 --num_workers 4 --optimizer sgd --beta 0.01
      ```
      | beta | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |------|----------------|------------|-----------------------------|
      | 0.01 | 69.8           | 30.4       | 1:34                        |
      | 0.05 | 70.8           | 29.4       | 1:34                        |
      | 0.09 | 72.9           | 32.6       | 1:34                        |
    
    - > B - Evaluate the performance of the Segmentation Network adapted with MBT.
      
      Multi-band Transfer consists in using the mean prediction of different segmentation networks trained with different spectral domain sizes (betas).

      ```bash
      main.py --dataset CROSS_DOMAIN --mode test_mbt --fda_b1_path trained_models\norm_fda0.01\best.pth --fda_b2_path trained_models\norm_fda0.05\best.pth --fda_b3_path trained_models\norm_fda0.09\best.pth
      ```

      Evaluate the model on the validation set of Cityscapes.

      | Accuracy _(%)_ | mIoU _(%)_ |
      |----------------|------------|
      | 73.1           | 33.0       |

    - > C - Implement a self-learning approach.

      Generate pseudo-labels for the training set of Cityscapes using the predictions of the model adapted with MBT. To avoid overfitting, filter out the low-confidence predictions.

      ```bash
      main.py --dataset CROSS_DOMAIN --data_transformation 0 --mode save_pseudo --fda_b1_path trained_models\norm_fda0.01\best.pth --fda_b2_path trained_models\norm_fda0.05\best.pth --fda_b3_path trained_models\norm_fda0.09\best.pth --save_pseudo_path dataset\Cityscapes\pseudo_label
      ```
      ![Alt text](images/image-2.png)

      Train the model with labeled synthetic data from GTA5 and pseudo-labeled real-world data from Cityscapes. Evaluate the model on the validation set of Cityscapes.
      
      ```bash
      main.py --dataset CROSS_DOMAIN --data_transformations 0 --batch_size 10 --learning_rate 0.01 --num_epochs 50 --save_model_path trained_models\selflearn_fda0.01 --resume False --comment selflearn_fda0.01 --mode self_learning --num_workers 4 --optimizer sgd --beta 0.01
      ```
      ```bash
      main.py --dataset CROSS_DOMAIN --data_transformations 0 --batch_size 10 --learning_rate 0.05 --num_epochs 50 --save_model_path trained_models\selflearn_fda0.05 --resume False --comment selflearn_fda0.05 --mode self_learning --num_workers 4 --optimizer sgd --beta 0.05
      ```
      ```bash
      main.py --dataset CROSS_DOMAIN --data_transformations 0 --batch_size 10 --learning_rate 0.01 --num_epochs 50 --save_model_path trained_models\selflearn_fda0.09 --resume False --comment selflearn_fda0.09 --mode self_learning --num_workers 4 --optimizer sgd --beta 0.09
      ```
      | beta | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |------|----------------|------------|-----------------------------|
      | 0.01 | 75.7           | 36.7       | 1:36                        |
      | 0.05 | 75.2           | 36.4       | 1:34                        |
      | 0.09 | 75.1           | 36.2       | 1:32                        |

    - > D - Evaluate the performance of the Segmentation Network trained with self-learning adapted with an additional step of MBT.

      Evaluate the model on the validation set of Cityscapes.

      ```bash
      main.py --dataset CROSS_DOMAIN --mode test_mbt --fda_b1_path trained_models\selflearn_fda0.01\best.pth --fda_b2_path trained_models\selflearn_fda0.05\best.pth --fda_b3_path trained_models\selflearn_fda0.09\best.pth
      ```

      | Accuracy _(%)_ | mIoU _(%)_ |
      |----------------|------------|
      | 75.9           | 37.5       |
   
