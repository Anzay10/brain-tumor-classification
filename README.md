# Brain Tumor Classification Using Deep Learning

This project presents a convolutional neural network (CNN)-based approach for classifying brain tumors from MRI images into four categories: glioma, meningioma, pituitary, and no tumor. The primary objective was to build an effective and interpretable deep learning pipeline that is both reproducible and expandable for future research. The study also includes a performance comparison with a pretrained MobileNetV2 model, as well as visual interpretability through Grad-CAM heatmaps.

## Project Objectives

The main goal of this project was to apply deep learning techniques to medical imaging and demonstrate their viability for tumor classification. Beyond achieving high classification performance, the project emphasizes transparency and interpretability by integrating Grad-CAM, a method that visualizes the regions in MRI images that influence the model’s decisions. This effort is aligned with broader goals of developing trustworthy and explainable AI models in healthcare.

## Dataset

The dataset used in this study is publicly available on Kaggle and contains T1-weighted contrast-enhanced MRI scans divided into four labeled classes: glioma, meningioma, pituitary tumor, and no tumor. The dataset includes images in JPEG format, pre-sorted into respective class folders. An 80:20 split was used to divide the dataset into training and validation sets. Data imbalance and subtle inter-class variations posed challenges during model training, which were partially addressed through augmentation and weighted loss functions.

## Methodology

The deep learning model developed is a custom convolutional neural network designed specifically for multi-class classification. The architecture consists of three convolutional blocks followed by batch normalization and max-pooling layers, with a dense classification head. The ReLU activation function is used for all intermediate layers, and the final output layer uses softmax activation, suitable for multi-class problems.

Images were resized to 128×128 pixels, and data augmentation was applied to training images using rotation, zoom, horizontal flips, shear, and pixel shifting to improve generalization. Early stopping was incorporated to prevent overfitting based on validation loss.

A separate experiment was conducted using MobileNetV2, a widely used pretrained architecture, to benchmark the performance of the custom CNN. The final model was saved using the `.h5` format for portability and reusability in other notebooks or projects, including potential extension to the BraTS dataset.

## Performance Results

The final CNN model achieved strong performance on the validation set. Key metrics included:

- Validation Accuracy: 83.7%
- Validation AUC: 0.964
- Validation Precision: 0.870
- Validation Recall: 0.809
- Validation Loss: 0.455

These results indicate a well-generalized model with a good balance between sensitivity and specificity. Performance was further analyzed using confusion matrices and class-wise metrics. While classification of glioma and meningioma remained more difficult, the model showed consistent improvement across all classes with the addition of data augmentation and class weighting.

The MobileNetV2 model underperformed relative to the custom CNN, suggesting that feature representations learned on natural images do not translate effectively to grayscale medical imaging tasks without significant fine-tuning.

## Model Interpretability

Grad-CAM was used to generate activation heatmaps that highlight regions in the input MRI scans that most influenced the network’s predictions. These visualizations confirm that the model consistently attends to tumor regions in correctly classified images. The use of Grad-CAM adds transparency to the model's decision-making process, an essential requirement for medical AI applications.

## Limitations and Challenges

Despite promising results, the model faced several limitations. Class imbalance and visually similar features between tumor types (e.g., meningioma and glioma) occasionally led to misclassifications. Additionally, data preprocessing and labeling inconsistencies in the original dataset presented challenges that required manual intervention. While the model generalizes well to validation data, performance on external datasets remains to be validated.

## Future Work

The project can be extended in several directions. The most immediate opportunity involves applying the trained model to the BraTS dataset, which contains 3D multi-modal scans with expert-annotated tumor segmentation. The saved model weights from this project can serve as a starting point for fine-tuning on BraTS or similar datasets. Other areas for exploration include using 3D CNN architectures, integrating attention mechanisms, or applying advanced uncertainty estimation methods to quantify prediction confidence.

## Repository Contents

- `brain_tumour.ipynb`: Main notebook with training, evaluation, Grad-CAM, and MobileNet comparison.
- `brain_tumour.py`: Script version of the model pipeline.
- `model_weights.h5`: Trained CNN model weights.
- `requirements.txt`: Required packages and dependencies.
- `README.md`: Project overview and documentation.

## Conclusion

This project demonstrates the application of deep learning for brain tumor classification using MRI scans. Through a custom CNN architecture and interpretability tools, it contributes to the development of explainable AI models for healthcare imaging. The model’s performance, interpretability, and extensibility make it a valuable foundation for further research and potential real-world deployment.

For inquiries or collaborations, please feel free to connect.

Anza Shoaib  
Teaching Fellow, Lahore School of Economics  
anzashoaib2@gmail.com

