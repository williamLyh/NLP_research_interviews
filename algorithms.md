

# Binary Search Tree (BST)

- Huffman Tree

# Algorithms

## SVM
- C penalty constant: Control the tolerance to the mis-classified items. 惩罚系数,控制模型对误分类的“容忍度”
  - 增大C参数会增加模型的复杂度，因为它增加了对误分类的惩罚，使得模型更倾向于划分更复杂的边界。
  - 减小C参数会减少模型对误分类的惩罚，可能会减少过拟合。
- Gamma kernal func constant: Control the impact of single sample to the decision boundary. 核函数系数，控制单个样本对模型决策边界的影响范围（主要用于RBF核）
  - 增大gamma参数会增加模型的复杂度，因为它控制了每个训练样本的影响范围，较大的gamma值会导致更复杂的决策边界。
  - gamma越大，核函数的宽度就越小，模型的复杂度就越高，可能会导致过拟合。
