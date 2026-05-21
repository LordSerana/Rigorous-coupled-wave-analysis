function [ RVec, real_Ray1_idx, real_Ray2_idx, m1, m2, B1, B2, b0 ] = Make_Computation( Lam, eps1, eps2, plot_fields, n2 )
% Makes the whole computation
%   Inputs are wavelength and polarization of the
%   grating and the light, output is the vector of reflected and
%   transmiited amplitudes
[Switches, Parameters]=setParameters();
%% Set constants

d=Parameters(1).ParamF(1);                   %period of the grating            
thI=Parameters(1).ParamF(2)*pi/180;          %incident angle
n1=Parameters(1).ParamF(3);                  %refraction index of incident medium
K=2*pi/d;                  
nTr=Parameters(1).ParamF(4);                 %Truncation number
nDim=2*nTr+1;                               %Total number of modes
h=Parameters(1).ParamF(6);                   %Depth of the grating

if (h/d>1.4)
    disp('Deep grating, results can be inaccurate')
end
%% Set wavenumber
k0=2*pi/Lam;                                %wavenumber

%% Grating profile and its derivative

a_fun=@(x) k0.*(Parameters(1).Prof(x./k0));
a_diff_fun=@(x) Parameters(2).Prof(x./k0);

%% Decide whether to cut small array elements
cut_small=1;

if cut_small
    tol=Parameters(1).ParamF(5);             %error tolerance
else
    tol=0;
end

%% Auxiliary constants

alfa0=n1*k0*sin(thI);
m1=-floor(alfa0/K)-(nDim-1)/2;     %lower order for adaptive truncation
m2=-floor(alfa0/K)+(nDim-1)/2;     %upper order for adaptive truncation

%% Prepare fields
nDim=m2-m1+1;                       %number of modes
alfa0=n1*k0*sin(thI);               %alpha0 from (3)
A=(alfa0*ones(1,m2-m1+1)+K.*linspace(m1,m2,nDim))./k0; %alpha field from (2)

B1=n1^2-A.^2;                       %beta_1^2 field from (4)
B2=n2^2-A.^2;                       %beta_2^2 field from (4)

if (min(abs(B1)<tol))               %When some element of B1 is equal to zero, break
    disp('System resonance, change the incident angle slightly');
    return;
end
if (min(abs(B2)<tol))               %When some element of B2 is equal to zero, break
    disp('System resonance, change the incident angle slightly')
    return;
end
SB1=sqrt(B1);                       %Eigenvalues of the original problem, incident medium
SB1_idx=(abs(imag(SB1))==0)&(real(SB1)>0); %indices of positive real propagation orders, incident medium
SB1_ind=(m1:m2);                    %indices of all modes
real_Ray1_idx=SB1_ind(SB1_idx);     %indices of real propagation numbers, positive due to radiation conditions

SB2=-sqrt(B2);                      %Eigenvalues of the original problem, incident medium
SB2_idx2=(abs(imag(SB2))==0)&(real(SB2)<0); %indices of negative real propagation orders, transmission medium
SB2_ind2=(m1:m2);                   %indices of all modes
real_Ray2_idx=SB2_ind2(SB2_idx2);   %indices of all modes

%% Evaluate the FFT of function a_diff
a_diff_vec=F_series_gen(a_diff_fun,12,k0*d,nDim);
a_mat=toeplitz(a_diff_vec);

%% Get eigenmodes and eigenvectors
[V1,rho1,V2,rho2]=EigChand(A,B1,B2,a_mat,nDim);
%If the script has problems with a lack of memory, delete superfluous variables
%clear ChandM1 ChandM2 BB1 BB2

%% Sort eigenvalues
[real_eig1p,real_eig2n,imag_eig1p,imag_eig2n,imag_Vec1p,imag_Vec2n]=SortEigenvaluesChand(rho1,V1,rho2,V2,tol,nDim);
%If the script has problems with a lack of memory, delete superfluous variables
%clear rho1 rho2 V1 V2

%% Assemble F-matrices
b0=sqrt(B1(1-m1));

[F_in,FRN,FRP]=GenerateFFieldsChand(a_fun,b0,nDim,k0,d,m1,m2,real_Ray2_idx,real_Ray1_idx,SB1,SB2);

%% Assemble G matrices
[G_RP,G_RN,G_in,G_P,G_N]=GenerateGFieldsChand(b0,a_mat,real_eig1p,real_eig2n,SB1,SB2,real_Ray1_idx,real_Ray2_idx,m1,m2,nDim,imag_eig1p,imag_eig2n,FRP,FRN,F_in,imag_Vec1p,imag_Vec2n,A,eps1,eps2);

%% Assemble matrix for matching boundary conditions and solve the linear system
MatBC=[FRP imag_Vec1p -FRN -imag_Vec2n;G_RP G_P -G_RN -G_N];
VecBC=[-F_in; -G_in];
RVec=MatBC\VecBC;

%% Plot intensity if required
if(plot_fields)
    save_plot=Switches(3).PlotFields;                   %Name for saving the plot
    Plot_Intensity( d,k0,h,alfa0,b0,a_fun,RVec,real_Ray1_idx, SB1, A, real_Ray2_idx, real_eig1p, imag_Vec1p, imag_eig1p, SB2, real_eig2n, imag_Vec2n,imag_eig2n, m1, m2, save_plot, Lam);
end