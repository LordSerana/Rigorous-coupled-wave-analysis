%% Generate G_R^+, G_R^- matrices from the output
function [G_RP,G_RN,G_in,G_P,G_N]=GenerateGFieldsChand(b0,a_mat,real_eig1p,real_eig2n,SB1,SB2,real_Ray1_idx,real_Ray2_idx,m1,m2,nDim,imag_eig1p,imag_eig2n,FRP,FRN,F_in,imag_Vec1p,imag_Vec2n,A,eps1,eps2)
G_RP=zeros(nDim,length(real_eig1p));
G_RN=zeros(nDim,length(real_eig2n));
G_in=zeros(nDim,1);
AUX_mat=(eye(nDim) + a_mat^2);
for M=1:nDim
    for N=1:length(real_eig1p)
        for NN=1:nDim
            G_RP(M,N)=G_RP(M,N)+ (a_mat(M,NN)*A(NN) - AUX_mat(M,NN)*SB1(N+min(real_Ray1_idx)-m1))*FRP(NN,N); %to do the sum used two for cycles
        end
    end
end

for M=1:nDim
    for N=1:length(real_eig2n)
        for NN=1:nDim
           G_RN(M,N)=G_RN(M,N)+ (a_mat(M,NN)*A(NN) - AUX_mat(M,NN)*SB2(N+min(real_Ray2_idx)-m1))*FRN(NN,N);
        end
    end
end

for M=1:nDim
    for NN=1:nDim
        G_in(M)=G_in(M)+ (a_mat(M,NN)*A(NN) + AUX_mat(M,NN)*b0)*F_in(NN);
    end
end

G_P=zeros(nDim,length(imag_eig1p));
G_N=zeros(nDim,length(imag_eig2n));
for M=1:nDim
    for N=1:length(imag_eig1p)
        for NN=1:nDim
            G_P(M,N)=G_P(M,N)+ (a_mat(M,NN)*A(NN) - AUX_mat(M,NN)*(imag_eig1p(N)))*imag_Vec1p(NN,N);
        end
    end
end

for M=1:nDim
    for N=1:length(imag_eig2n)
        for NN=1:nDim
            G_N(M,N)=G_N(M,N)+ (a_mat(M,NN)*A(NN) - AUX_mat(M,NN)*(imag_eig2n(N)))*imag_Vec2n(NN,N);
        end
    end
end

G_RP=(1/eps1).*G_RP; %  Normalizing by permittivities
G_RN=(1/eps2).*G_RN;
G_in=(1/eps1).*G_in;
G_P=(1/eps1).*G_P;
G_N=(1/eps2).*G_N;