 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load the dataset
df = pd.read_csv('real_estate_dataset.csv')

#Print the first few rows of the dataset
columns = df.columns

print(f'Columns in the dataset : {columns}')
np.savetxt('columns.txt', columns, fmt='%s', delimiter=',')

#Use Square_Feet, Garage_Size, Location_Score, Distance_to_Center as features
X = df[['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center']].values
y = df['Price'].values

print(f'Shape of X: {X.shape}')
print(f'data type of X: {X.dtype}')

#Calculate the number of samples and features
n_samples, n_features = X.shape

#Append a column of ones to X 
coefs = np.ones(n_features + 1)
print (f'Shape of coefs : {coefs.shape}')

#predict the price for each sample in X
predictions_bydefn = np.dot(X, coefs[1:]) + coefs[0]
 
#append a column of ones to X
X = np.hstack([np.ones((n_samples, 1)), X])
 
#predict the price for each sample in X
predictions = X @ coefs

#Check if the predictions are the same
is_same = np.allclose(predictions, predictions_bydefn)
print(f'Are the predictions the same? {is_same}')

#Calculate the errors using the initial coefficients
errors = predictions - y
 
print(f'Size of errors : {errors.shape}')
print(f'L2 Norm of errors : {np.linalg.norm(errors)}')
 
#Calculate the relative errors
rel_errors = errors / y
print(f'L2 Norm of relative errors : {np.linalg.norm(rel_errors)}')

 
#Calculate the mead of sqaurre of errors
loss_loop = 0
for i in range(n_samples):
    loss_loop += (predictions[i] - y[i])**2

loss_loop /= n_samples

 
loss_matrix = np.transpose(errors) @ errors / n_samples

#Check if the losses are the same
is_diff = np.allclose(loss_loop, loss_matrix)
print(f'Are the losses the same? {is_diff}')


#Objective function : f(coefs) = 1/n_samples * ||X @ coefs - y||^2
#Gradient of f(coefs) = 2/n_samples * X^T @ (X @ coefs - y

#What is a soltion?
#A solution is a set of coefficients that minimizes the objective function

#How do we find a solution?
#By searching for the coefficients at which the gradient is zero
# Or I can set the gradient to zero and solve for the coefficients

#Write the loss matrix in the terms of data and coefs
loss_matrix = (X @ coefs - y).T @ (X @ coefs - y) / n_samples

 
#Calculate the gradient of the loss with respect to the coefficients
gradient = 2/n_samples * X.T @ (X @ coefs - y)

 
#we set grad_matrix = 0 and solve for coefs
#X^T @ X @ coefs = X^T @ y. This is called Normal Equation
#coefs = (X^T @ X)^-1 @ X^T @ y

coefs = np.linalg.inv(X.T @ X) @ X.T @ y

#Save the coefficients to a file
np.savetxt('coefs.txt', coefs, fmt='%f', delimiter=',')

 
predictions = X @ coefs

 
#Calculate the errors using the optimal coefficients
errors = predictions - y

 
#print the L2 norm of the errors
print(f'L2 Norm of errors_model : {np.linalg.norm(errors)}')

 
relatve_errors = errors / y
print(f'L2 Norm of relative errors_model : {np.linalg.norm(relatve_errors)}')

 
#Use all the features in the dataset to build a linear model
X = df.drop('Price', axis=1).values
y = df['Price'].values

n_samples, n_features = X.shape
print(f'number of samples and features : {n_samples, n_features}')

 
X = np.hstack([np.ones((n_samples, 1)), X])
coefs = np.linalg.inv(X.T @ X) @ X.T @ y

 
#Save the coefficients to a file
np.savetxt('coefs_all.csv', coefs, fmt='%f', delimiter=',')

 
#Calculate the rank of X^T @ X
rank = np.linalg.matrix_rank(X.T @ X)
print(f'Rank of X^T @ X : {rank}')

 
#Solve the normal equation using matrix decomposition
#QR decomposition
Q, R = np.linalg.qr(X)

print(f'Shape of Q : {Q.shape}')
print(f'Shape of R : {R.shape}')

 
np.savetxt('R.csv', R, fmt='%f', delimiter=',')

 
sol = Q.T @ Q
np.savetxt('sol.csv', sol, fmt='%f', delimiter=',')

 
#R*coefs = b

#X = QR
#X^T @ X = R^T @ Q^T @ Q @ R = R^T @ R
#X^T @ y = R^T @ Q^T @ y
#R @ coefs = Q^T @ y

b = Q.T @ y

print(f'Shape of b : {b.shape}')
print(f'Shape of R : {R.shape}')

coefs_qr = np.linalg.inv(R) @ b
# loop to solve for R @ coefs = b using back substitution

coefs_qr_loop = np.zeros(n_features + 1)
for i in range(n_features, -1, -1):
    coefs_qr_loop[i] = b[i]
    for j in range(i + 1, n_features + 1):
        coefs_qr_loop[i] -= R[i, j] * coefs_qr_loop[j]
    coefs_qr_loop[i] /= R[i, i]

#Check if the coefficients are the same
is_same = np.allclose(coefs_qr, coefs_qr_loop)
print(f'Are the coefficients the same with qr loop and qr library? {is_same}')
is_same = np.allclose(coefs_qr, coefs)
print(f'Are the coefficients the same with qr loop and coefs ? {is_same}')

np.savetxt('coefs_qr.csv', coefs_qr_loop, fmt='%f', delimiter=',')

 
#Solving the normal equation using SVD
#X = U @ S @ V^T
#X^-1 =  
U, S, Vt = np.linalg.svd(X, full_matrices=False)

#Homework
 
#Eigen decomposition of square matrix
#A = V @ D @ V^-1
#A^-1 = V @ D^-1 @ V^-1
#A = X^T @ X -> symmetric square matrix
#A = V @ D @ V^T , A^-1 = V @ D^-1 @ V^T
#

#X @ coefs = y
#Normal Equation : X^T @ X @ coefs = X^T @ y

#Find inverse of X in least squares sense
#Pseudo inverse of X
#Xdagger = (X^T @ X)^-1 @ X^T

 
#To complete: Calculate the coefs_svd using the SVD decomposition

#X = U @ S @ V^T
#X^T @ X = V @ S^2 @ V^T
#X^T @ y = V @ S @ U^T @ y
#coefs_svd = V @ S^-1 @ U^T @ y

U, S, Vt = np.linalg.svd(X, full_matrices=False)

coefs_svd = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y

np.savetxt('coefs_svd.csv', coefs_svd, fmt='%f', delimiter=',')

#Check if the coefficients are the same
is_same = np.allclose(coefs_svd, coefs)
print(f'Are the coefficients the same svd decomposition? {is_same}')

 
#Calculate the predictions using the eigen decomposition of X^T @ X
#X^T @ X = V @ D @ V^T
#Normal Equation : X^T @ X @ coefs = X^T @ y
#Calculate the coefs using the eigen decomposition of X^T @ X
d, v = np.linalg.eig(X.T @ X)
coefs_eigen = v @ np.linalg.inv(np.diag(d)) @ v.T @ X.T @ y

np.savetxt('coefs_eigen.csv', coefs_eigen, fmt='%f', delimiter=',')

#Check if the coefficients are the same
is_same = np.allclose(coefs_eigen, coefs)
print(f'Are the coefficients the same with eigen decomposition? {is_same}')


