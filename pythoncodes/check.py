import numpy as np
import scipy.linalg as la
J=np.array([[0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,1,1,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,1],
            [0,0,0,0,0,0,1]])
J1=np.array([[0,1,0],[0,0,0],[0,0,0]]) 
J2=np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]]) 

A1=np.matmul(J,J)
rank1=np.linalg.matrix_rank(A1)
print('Nullity of J^2:',7-rank1)
A2=np.matmul(J-np.eye(7),J-np.eye(7))
rank2=np.linalg.matrix_rank(A2)
print('Nullity of (J-I)^2:',7-rank2)
Vb1=la.null_space(A1)
Vb2=la.null_space(A2)
print('Basis for V1:\n',Vb1)
print('Basis for V2:\n',Vb2)
#print('T^2(T-I)^2:\n',np.matmul(A1,A2))
#print('T(T-I)^2:\n',np.matmul(J,A2))

#cyclic space 1
alpha1=np.array([[1],[1],[1],[1],[1],[1],[1]])
Ta1=np.matmul(J,alpha1)
T2a1=np.matmul(J,Ta1)
T3a1=np.matmul(J,T2a1)
b1=np.concatenate((alpha1.T,Ta1.T,T2a1.T,T3a1.T))
r1=np.linalg.matrix_rank(b1)
print('Basis vectors of space 1 are linearly independent because rank=',r1)

#cyclic space 2
alpha2=np.array([[0],[0],[1],[1],[1],[0],[0]])
#print('T(T-I)^2alpha2:\n',np.matmul(np.matmul(J,A2),alpha2))
Ta2=np.matmul(J,alpha2)
T2a2=np.matmul(J,Ta2)
b2=np.concatenate((alpha2.T,Ta2.T,T2a2.T))
r2=np.linalg.matrix_rank(b2)
print('Basis vectors of space 2 are linearly independent because rank=',r2)
total=np.concatenate((b1,b2))
print('Decomposition is possible because basis vectors matrix rank=',np.linalg.matrix_rank(total))
