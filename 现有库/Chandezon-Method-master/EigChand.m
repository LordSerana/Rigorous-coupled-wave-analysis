%% From the inputs A,B1,B2,a_mat,nDim assembles matrix of Chandezon method and returns eigenvalues and eigenvectors
function [V1, rho1, V2, rho2]  = EigChand(A,B1,B2,a_mat,nDim)

IB1=diag(B1)\eye(nDim);
IB2=diag(B2)\eye(nDim);
AUX_mat=(eye(nDim) + a_mat^2);

ChandM1=[-IB1*(diag(A)*a_mat + a_mat*diag(A))  (IB1*AUX_mat); eye(nDim),zeros(nDim)];     %eigenvalue matrix from (12), incident medium
ChandM2=[-IB2*(diag(A)*a_mat + a_mat*diag(A))  (IB2*AUX_mat); eye(nDim),zeros(nDim)];     %eigenvalue matrix from (12), transmission medium

[V1,rho1]=eig(ChandM1);             %Eigenvalues and eigenvectors of ChandM1
[V2,rho2]=eig(ChandM2);             %Eigenvalues and eigenvectors of ChandM2

rho1=1./diag(rho1);                 %Eigenvalues of ChandM1
rho2=1./diag(rho2);                 %Eigenvalues of ChandM2
