# Iris Classifier

A Random Forest-based classifier for the Iris dataset.

## Model Parameters

| Parameter       | Value |
|-----------------|-------|
| Algorithm       | Random Forest |
| n_estimators    | 100   |
| test_size       | 0.2   |
| random_state    | 42    |
| Target column   | species |

## Dataset

- **Source:** scikit-learn `load_iris()`
- **Samples:** 150
- **Features:** 4 (sepal length, sepal width, petal length, petal width)
- **Classes:** 3 (setosa, versicolor, virginica)

## Test Input

| Feature         | Value |
|-----------------|-------|
| Sepal Length    | 5.1   |
| Sepal Width     | 3.5   |
| Petal Length    | 1.4   |
| Petal Width     | 0.2   |
| **Expected**    | **setosa** |

## Setup

```bash
pip install scikit-learn pandas numpy
python main.py


#niranjan edited this file