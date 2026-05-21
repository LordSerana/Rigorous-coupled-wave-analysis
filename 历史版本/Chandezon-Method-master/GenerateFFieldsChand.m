%% Generates vectors F_in, F_R^+, F_R^- from the inputs
function [F_in,FRN,FRP]=GenerateFFieldsChand(a_fun,b0,nDim,k0,d,m1,m2,real_Ray2_idx,real_Ray1_idx,SB1,SB2)
LP_fun=@(x) exp(-(1i*b0).*a_fun(x)); %define fourier transform argument for positive fields
LN_fun=@(x) exp((1i*b0).*a_fun(x)); %define fourier transform argument for negative fields
F0=F_series_gen(LP_fun,12,k0*d,nDim); %Do FFT and extract coefficients of fourier fields
F1=F_series_gen(LN_fun,12,k0*d,nDim); %Do FFT and extract coefficients of fourier fields
F_in=[flip(F1(2:-m1+1));conj(F0(1:m2+1))]; %F_in vector: take first 1-m1 terms, and last m2+1 terms (butterfly algorithm)
FRN=zeros(nDim,length(real_Ray2_idx)); %preallocate fields
FRP=zeros(nDim,length(real_Ray1_idx)); %preallocate fields
          

for M=min(real_Ray1_idx):max(real_Ray1_idx)    
    LPn_fun=@(x) exp((1i*SB1(M-m1+1)).*a_fun(x)); %
    fm=12;      %exponent to number of fourier modes
    FRP0=F_series_gen(LPn_fun,fm,k0*d,2^fm);    %generate FFT of exponential of function profile, extract coefficients of fourier series
    FRP(:,M-min(real_Ray1_idx)+1)=[conj(FRP0((2^fm+m1+1-M):(2^fm)));conj(FRP0(1:(m2+1-M)))]; %Assmeble FRP vector: take first 1-m1 terms, and last m2+1 terms (butterfly algorithm)
end

%Do the same for fields in substrate region
for M=min(real_Ray2_idx):max(real_Ray2_idx)    
    LNn_fun=@(x) exp((1i*SB2(M-m1+1)).*a_fun(x));
    fm=12;%exponent to number of fourier modes
    FRN0=F_series_gen(LNn_fun,fm,k0*d,2^fm);
    FRN(:,M-min(real_Ray2_idx)+1)=[conj(FRN0((2^fm+m1+1-M):(2^fm)));conj(FRN0(1:(m2+1-M)))];
end