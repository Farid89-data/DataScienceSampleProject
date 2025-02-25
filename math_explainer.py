"""
Mathematical Concepts Explanation Module

This module provides explanations and demonstrations of the mathematical concepts
underlying the computer vision and machine learning techniques used in the project.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from pathlib import Path

from utils import ensure_dir_exists, save_text_to_file, timer

logger = logging.getLogger("ImageClassifier.MathExplainer")


class MathExplainer:
    """Class for generating mathematical explanations and demonstrations."""

    def __init__(self, config, data, model, evaluation_results):
        """
        Initialize the math explainer.

        Args:
            config: Configuration dictionary
            data: Dictionary containing data
            model: Trained Keras model
            evaluation_results: Dictionary containing evaluation results
        """
        self.config = config
        self.data = data
        self.model = model
        self.results = evaluation_results

        # Ensure output directories exist
        self.explanations_dir = Path(config['explanations_dir'])
        ensure_dir_exists(self.explanations_dir)
        self.plots_dir = Path(config['plots_dir'])
        ensure_dir_exists(self.plots_dir)

    @timer
    def explain_cnn_mathematics(self):
        """
        Generate explanations of the mathematical concepts in the CNN model.
        """
        logger.info("Generating comprehensive mathematical explanation of CNN concepts")

        # Create a detailed explanation document
        explanation = """
# Mathematical Explanation of Convolutional Neural Networks

This document provides a comprehensive explanation of the mathematical concepts underlying the CNN model used in this project.

## 1. Convolutional Layer Operations

### 1.1 Convolution Operation

The 2D convolution operation in our model can be expressed mathematically as:

$$(I * K)_{i,j} = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I_{i+m,j+n} \cdot K_{m,n}$$

Where:
- $I$ is the input feature map (or image)
- $K$ is the convolutional kernel/filter of size $k_h \times k_w$
- $*$ denotes the convolution operation

For a convolutional layer with multiple input channels and filters, the operation becomes:

$$(I * K)_{i,j,d'} = \sum_{c=0}^{C-1} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I_{i+m,j+n,c} \cdot K_{m,n,c,d'}$$

Where:
- $C$ is the number of input channels
- $d'$ is the output channel (filter) index

### 1.2 Activation Function (ReLU)

After convolution, we apply the Rectified Linear Unit (ReLU) activation function:

$$f(x) = \max(0, x)$$

This introduces non-linearity into our model while being computationally efficient. The gradient of ReLU is:

$$f'(x) = \begin{cases} 
0 & \text{for } x < 0 \\
1 & \text{for } x > 0
\end{cases}$$

## 2. Batch Normalization

Batch normalization normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation:

$$\hat{x}^{(k)} = \frac{x^{(k)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

Then scales and shifts the result:

$$y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$

Where:
- $\mu_B = \frac{1}{m}\sum_{i=1}^{m}x^{(i)}$ is the batch mean
- $\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x^{(i)} - \mu_B)^2$ is the batch variance
- $\epsilon$ is a small constant for numerical stability
- $\gamma$ and $\beta$ are learnable parameters

During inference, running statistics are used:

$$\hat{x}^{(k)} = \frac{x^{(k)} - E[x]}{\sqrt{Var[x] + \epsilon}}$$

## 3. Pooling Operations (MaxPooling)

MaxPooling returns the maximum value from the portion of the feature map covered by the kernel:

$$\text{MaxPool}(X)_{i,j} = \max_{m \in [0,k_h), n \in [0,k_w)} X_{s_h \cdot i + m, s_w \cdot j + n}$$

Where:
- $k_h, k_w$ are the pool sizes
- $s_h, s_w$ are the strides
- $X$ is the input feature map

## 4. Residual Connections

Residual connections help mitigate the vanishing gradient problem by allowing the gradient to flow directly through the network:

$$y = F(x, W) + x$$

Where:
- $F(x, W)$ represents the residual mapping to be learned
- $x$ is the identity mapping (skip connection)
- $y$ is the output of the residual block

## 5. Linear Algebra in Fully Connected Layers

After flattening, we apply fully connected layers which perform matrix multiplication:

$$Y = W \cdot X + b$$

Where:
- $X$ is the input vector (or matrix for batch processing)
- $W$ is the weight matrix where $W_{i,j}$ connects input unit $j$ to output unit $i$
- $b$ is the bias vector
- $Y$ is the output vector

## 6. Softmax Function for Classification

In the final layer, we use the softmax function to convert the raw scores to probabilities:

$$\text{Softmax}(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$$

Where:
- $z_j$ is the raw score for class $j$
- $K$ is the number of classes

The softmax ensures that all outputs are in the range [0, 1] and sum to 1.

## 7. Loss Function: Sparse Categorical Cross-Entropy

We use sparse categorical cross-entropy loss:

$$\text{Loss} = -\frac{1}{N}\sum_{i=1}^{N} \log(p_{i,y_i})$$

Where:
- $p_{i,y_i}$ is the predicted probability for the true class $y_i$ of the $i$-th sample
- $N$ is the number of samples

## 8. Backpropagation and Gradient Descent

The backpropagation algorithm computes the gradient of the loss function with respect to the weights using the chain rule:

$$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(l+1)}} \cdot \frac{\partial a^{(l+1)}}{\partial z^{(l+1)}} \cdot \frac{\partial z^{(l+1)}}{\partial W^{(l)}}$$

Where:
- $L$ is the loss function
- $W^{(l)}$ are the weights of layer $l$
- $z^{(l)}$ is the weighted input of layer $l$
- $a^{(l)}$ is the activation output of layer $l$

## 9. Optimization with Adam

For optimization, we use the Adam optimizer, which combines ideas from momentum and RMSprop:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

Where:
- $g_t$ is the gradient at time step $t$
- $m_t$ is the first moment estimate (mean of gradients)
- $v_t$ is the second moment estimate (uncentered variance of gradients)
- $\beta_1$ and $\beta_2$ are hyperparameters (typically 0.9 and 0.999)

The bias-corrected moments are:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The parameter update rule is:

$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where:
- $\alpha$ is the learning rate
- $\epsilon$ is a small constant for numerical stability

## 10. Regularization Techniques

### 10.1 Dropout

Dropout randomly sets a fraction of the input units to 0 during training:

$$\text{Dropout}(x) = \begin{cases} 
\frac{x}{1-p} & \text{with probability } 1-p \\
0 & \text{with probability } p
\end{cases}$$

Where $p$ is the dropout rate. The scaling by $\frac{1}{1-p}$ ensures that the expected sum of the inputs remains the same.

### 10.2 L2 Regularization

L2 regularization adds a penalty term to the loss function:

$$L_{reg} = L + \frac{\lambda}{2} \sum_{l} \sum_{i} \sum_{j} (W_{i,j}^{(l)})^2$$

Where:
- $L$ is the original loss function
- $\lambda$ is the regularization strength
- $W_{i,j}^{(l)}$ is the weight at position $(i,j)$ in layer $l$

## 11. Evaluation Metrics

### 11.1 Accuracy

$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$$

### 11.2 Precision and Recall

$$\text{Precision}_i = \frac{TP_i}{TP_i + FP_i}$$

$$\text{Recall}_i = \frac{TP_i}{TP_i + FN_i}$$

Where for class $i$:
- $TP_i$ (True Positives): Correctly predicted as class $i$
- $FP_i$ (False Positives): Incorrectly predicted as class $i$
- $FN_i$ (False Negatives): Incorrectly predicted as not class $i$

### 11.3 F1 Score

The F1 score is the harmonic mean of precision and recall:

$$\text{F1}_i = 2 \cdot \frac{\text{Precision}_i \cdot \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$$

## 12. Data Augmentation

Data augmentation applies transformations $T$ to training data $X$ to create new training examples:

$$X_{aug} = T(X)$$

Common transformations include:
- Rotation: $T_{rot}(X, \theta) = R_{\theta} \cdot X$, where $R_{\theta}$ is a rotation matrix
- Translation: $T_{trans}(X, \Delta x, \Delta y) = X + [\Delta x, \Delta y]^T$
- Scaling: $T_{scale}(X, s_x, s_y) = S \cdot X$, where $S$ is a scaling matrix
- Flipping: $T_{flip}(X) = F \cdot X$, where $F$ is a reflection matrix
"""

        # Save the explanation to a file
        save_text_to_file(
            explanation,
            self.explanations_dir / 'cnn_mathematical_explanation.md'
        )

        logger.info("CNN mathematical explanation generated and saved")

    @timer
    def demonstrate_eigenvectors(self):
        """
        Demonstrate understanding of eigenvectors and eigenvalues
        through feature analysis.
        """
        logger.info("Demonstrating eigenvector computation and analysis")

        # Extract features from a subset of images
        num_samples = min(1000, len(self.data['X_train']))
        # Flatten the images and select a subset of pixels as features
        flattened = self.data['X_train'][:num_samples].reshape(num_samples, -1)

        # Compute correlation matrix
        corr_matrix = np.corrcoef(flattened, rowvar=False)

        # Select a smaller subset for visualization (first 10 features)
        small_corr = corr_matrix[:10, :10]

        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(small_corr)

        # Sort by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Plot correlation matrix
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        sns.heatmap(small_corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Feature Correlation Matrix', fontweight='bold')

        # Plot eigenvalues
        plt.subplot(2, 2, 2)
        plt.bar(range(len(eigenvalues)), eigenvalues)
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalues of Correlation Matrix', fontweight='bold')
        plt.xticks(range(len(eigenvalues)))
        plt.grid(True, alpha=0.3)

        # Plot first eigenvector
        plt.subplot(2, 2, 3)
        plt.bar(range(len(eigenvectors[:, 0])), eigenvectors[:, 0])
        plt.xlabel('Feature Index')
        plt.ylabel('Weight')
        plt.title('First Eigenvector (Principal Component)', fontweight='bold')
        plt.xticks(range(len(eigenvectors[:, 0])))
        plt.grid(True, alpha=0.3)

        # Plot second eigenvector
        plt.subplot(2, 2, 4)
        plt.bar(range(len(eigenvectors[:, 1])), eigenvectors[:, 1])
        plt.xlabel('Feature Index')
        plt.ylabel('Weight')
        plt.title('Second Eigenvector', fontweight='bold')
        plt.xticks(range(len(eigenvectors[:, 1])))
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'eigenvector_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Display the first eigenvector
        logger.info(f"First eigenvector: {eigenvectors[:, 0]}")
        logger.info(f"Corresponding eigenvalue: {eigenvalues[0]}")

        # Save detailed explanation
        explanation = """
# Eigenvectors and Eigenvalues in Data Analysis

This document demonstrates the application of eigenvectors and eigenvalues in the context of image data analysis.

## 1. Mathematical Definition

For a square matrix A, an eigenvector v and corresponding eigenvalue λ satisfy:

$$A \\cdot v = \\lambda \\cdot v$$

This means that when A operates on v, it scales v by a factor of λ without changing its direction.

## 2. Significance in Data Analysis

### 2.1 Principal Component Analysis (PCA)

Eigenvectors and eigenvalues are fundamental to PCA, which is a dimensionality reduction technique. In PCA:

1. We compute the covariance/correlation matrix of our features
2. We find the eigenvectors and eigenvalues of this matrix
3. The eigenvectors represent the principal components - directions of maximum variance
4. The eigenvalues indicate how much variance is explained by each principal component
5. By selecting eigenvectors with the largest eigenvalues, we can reduce dimensionality while preserving maximum variance

### 2.2 Geometric Interpretation

- Eigenvectors represent directions in feature space that remain unchanged by the linear transformation (except for scaling)
- Eigenvalues represent the amount of stretch or compression along these directions
- The eigenvector with the largest eigenvalue points in the direction of maximum variance

## 3. Application to Image Data

In our project, we:
1. Flattened a subset of images into feature vectors
2. Computed the correlation matrix between features
3. Extracted eigenvectors and eigenvalues
4. Identified the principal components

This analysis reveals:
- How features (pixels) correlate with each other
- Which linear combinations of features capture the most variance
- Potential directions for dimensionality reduction

## 4. Mathematical Derivation

For a dataset X with n samples and d features, the covariance matrix Σ is:

$$\\Sigma = \\frac{1}{n} \\sum_{i=1}^{n} (x_i - \\mu)(x_i - \\mu)^T = \\frac{1}{n}X^TX$$

where μ is the mean vector and X is mean-centered.

The correlation matrix normalizes the covariance:

$$R_{i,j} = \\frac{\\Sigma_{i,j}}{\\sigma_i \\sigma_j}$$

To find eigenvectors and eigenvalues, we solve:

$$|A - \\lambda I| = 0$$

This gives us eigenvalues λ, and for each λ, we find eigenvectors v by solving:

$$(A - \\lambda I)v = 0$$

## 5. Real-World Applications

- **Image Compression**: Representing images with fewer principal components
- **Facial Recognition**: Eigenfaces approach
- **Noise Reduction**: Removing components with small eigenvalues
- **Feature Extraction**: Creating new, uncorrelated features
- **Data Visualization**: Projecting high-dimensional data to 2D or 3D

## 6. Connection to Neural Networks

- CNNs implicitly learn feature extractors that may align with principal components
- Understanding feature correlations helps in designing better architectures
- Singular Value Decomposition (related to eigendecomposition) can be used to compress network weights
"""

        # Save the explanation to a file
        save_text_to_file(
            explanation,
            self.explanations_dir / 'eigenvectors_explanation.md'
        )

        logger.info("Eigenvector demonstration and explanation completed")

    @timer
    def demonstrate_bayes_theorem(self):
        """
        Demonstrate understanding of Bayes' theorem in the context of classification.
        """
        logger.info("Demonstrating Bayes' theorem application to classification")

        # Get predictions and true labels
        y_pred = np.array(self.results['y_pred'])
        y_pred_classes = np.array(self.results['y_pred_classes'])
        y_true = np.array(self.results['y_true'])

        # Calculate prior probabilities P(C) for each class
        class_priors = {}
        total_samples = len(y_true)
        for c in range(len(self.data['class_names'])):
            class_count = np.sum(y_true == c)
            class_priors[c] = class_count / total_samples

        # Calculate likelihood P(x|C) and posterior P(C|x) for a few test examples
        num_examples = 5
        bayes_examples = []

        # Select diverse examples (some correct, some incorrect predictions)
        correct_indices = np.where(y_pred_classes == y_true)[0]
        incorrect_indices = np.where(y_pred_classes != y_true)[0]

        # Select 3 correct and 2 incorrect examples if available
        selected_indices = []
        if len(correct_indices) >= 3:
            selected_indices.extend(np.random.choice(correct_indices, 3, replace=False))
        else:
            selected_indices.extend(correct_indices)

        if len(incorrect_indices) >= 2:
            selected_indices.extend(
                np.random.choice(incorrect_indices, min(2, 5 - len(selected_indices)), replace=False))
        else:
            selected_indices.extend(incorrect_indices)

        # If we still need more examples, select randomly
        if len(selected_indices) < num_examples:
            remaining = num_examples - len(selected_indices)
            all_indices = np.arange(len(y_true))
            mask = np.ones(len(y_true), dtype=bool)
            mask[selected_indices] = False
            remaining_indices = all_indices[mask]
            selected_indices.extend(np.random.choice(remaining_indices, remaining, replace=False))

        # Process selected examples
        for idx in selected_indices[:num_examples]:
            true_class = int(y_true[idx])
            pred_class = int(y_pred_classes[idx])
            pred_probabilities = y_pred[idx]

            # Store the information
            bayes_examples.append({
                'index': int(idx),
                'image': self.data['X_test'][idx],
                'true_class': self.data['class_names'][true_class],
                'pred_class': self.data['class_names'][pred_class],
                'true_class_idx': true_class,
                'pred_probabilities': pred_probabilities,
                'class_priors': class_priors
            })

        # Visualize the examples with Bayesian interpretation
        plt.figure(figsize=(15, 12))

        for i, example in enumerate(bayes_examples):
            # Plot the image
            plt.subplot(num_examples, 2, 2 * i + 1)
            plt.imshow(example['image'])
            title = f"True: {example['true_class']}, Pred: {example['pred_class']}"
            # Color based on correctness
            color = 'green' if example['true_class'] == example['pred_class'] else 'red'
            plt.title(title, color=color, fontweight='bold')
            plt.axis('off')

            # Plot the probability distribution
            plt.subplot(num_examples, 2, 2 * i + 2)

            # Plot posteriors
            bars = plt.bar(
                self.data['class_names'],
                example['pred_probabilities'],
                alpha=0.7,
                label='Posterior P(C|x)'
            )

            # Highlight the true class
            bars[example['true_class_idx']].set_color('green')
            bars[example['true_class_idx']].set_hatch('///')

            # Plot priors (smaller bars)
            prior_values = [example['class_priors'][j] for j in range(len(self.data['class_names']))]
            plt.bar(
                self.data['class_names'],
                prior_values,
                alpha=0.4,
                color='gray',
                width=0.4,
                label='Prior P(C)'
            )

            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.ylabel('Probability')
            plt.title(f"Class Probabilities for Example {i + 1}", fontweight='bold')
            plt.legend()
            plt.tight_layout()

        plt.savefig(self.plots_dir / 'bayes_theorem_examples.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save a detailed explanation
        explanation = """
# Bayes' Theorem in Image Classification

This document explains Bayes' theorem and its application to image classification in our CNN model.

## 1. Bayes' Theorem Formula

Bayes' theorem states:

$$P(C|X) = \\frac{P(X|C) \\cdot P(C)}{P(X)}$$

Where:
- $P(C|X)$ is the posterior probability: probability of class C given features X
- $P(X|C)$ is the likelihood: probability of features X given class C
- $P(C)$ is the prior probability of class C
- $P(X)$ is the evidence: total probability of features X

## 2. Components of Bayes' Theorem in Classification

### 2.1 Prior Probability $P(C)$

The prior probability $P(C)$ represents our belief about class distribution before seeing any data. In a balanced dataset, each class might have an equal prior. In an imbalanced dataset, the prior reflects the class imbalance.

For example, if 20% of our dataset consists of class 'cat', then:
$P(C=\\text{cat}) = 0.2$

### 2.2 Likelihood $P(X|C)$

The likelihood $P(X|C)$ is the probability of observing features X given that the instance belongs to class C. In a CNN, the network implicitly learns to model this likelihood through its convolutional and fully connected layers.

### 2.3 Evidence $P(X)$

The evidence $P(X)$ is the total probability of observing features X across all possible classes:

$$P(X) = \\sum_{c} P(X|c) \\cdot P(c)$$

It acts as a normalizing constant to ensure the posteriors sum to 1.

### 2.4 Posterior Probability $P(C|X)$

The posterior probability $P(C|X)$ is what we're ultimately interested in: the probability of class C given the observed features X. The softmax output of our CNN represents these posterior probabilities.

## 3. Bayesian Interpretation of CNNs

While traditional CNNs are not explicitly Bayesian, we can interpret them in a Bayesian framework:

1. **Neural Network Parameters**: The weights and biases of the network are point estimates that maximize the likelihood of the training data.

2. **Final Layer**: The pre-softmax outputs can be interpreted as log-likelihoods.

3. **Softmax Function**: Converts these log-likelihoods to posterior probabilities.

4. **Classification Decision**: Selects the class with the highest posterior probability.

## 4. Example Analysis

In the visualizations, we show:
- The original image (input X)
- The prior probabilities P(C) based on class distribution
- The posterior probabilities P(C|X) output by the model

The difference between the prior and posterior distributions shows how the model updates its belief after observing the image features.

## 5. Advantages of Bayesian Thinking

A Bayesian perspective offers several advantages:

1. **Uncertainty Quantification**: The posterior distribution provides a measure of confidence in predictions.

2. **Prior Knowledge Integration**: We can incorporate domain knowledge through priors.

3. **Decision Theory**: Makes the connection between probabilities and optimal decisions explicit.

4. **Handling Imbalanced Data**: Class imbalance is naturally addressed through priors.

## 6. Bayesian Neural Networks

True Bayesian Neural Networks go beyond our current model by:

1. **Parameter Uncertainty**: Representing weights and biases as probability distributions rather than point estimates.

2. **Posterior Inference**: Using methods like variational inference or MCMC to approximate the posterior over weights.

3. **Prediction Uncertainty**: Producing predictive distributions that capture both aleatoric uncertainty (data noise) and epistemic uncertainty (model uncertainty).

## 7. Mathematical Connection to Cross-Entropy Loss

The cross-entropy loss we use for training is directly related to the principle of maximum likelihood estimation in a Bayesian framework:

$$\\text{Loss} = -\\sum_{i=1}^{N} \\log(p_{i,y_i})$$

This is equivalent to minimizing the negative log-likelihood, which is a key step in finding the maximum a posteriori (MAP) estimate in Bayesian inference.

## 8. Practical Implications

Understanding the Bayesian interpretation of our CNN model helps us:

- Interpret prediction confidences properly
- Make better decisions in uncertain cases
- Design appropriate decision thresholds
- Recognize limitations in unusual or out-of-distribution cases
"""

        # Save the explanation to a file
        save_text_to_file(
            explanation,
            self.explanations_dir / 'bayes_theorem_explanation.md'
        )

        logger.info("Bayes theorem demonstration and explanation completed")