%% Matlab implementation of C-Method; Main File, parameters are loaded from the file setParameters.m

% Based on the paper from Li, Chandezon, Granet, Pluey: 
% Rigorous and efficient grating-analysis method made easy for optical
% engineers (1999)

clear all; clc;
%% Prepare constants and load parameters

[Switches, Parameters]=setParameters();
VecWaveLength=Parameters(1).ParamVec(1,:);
RefVec=zeros(length(VecWaveLength),1);
TranVec=zeros(length(VecWaveLength),1);
WaveLengthToPlot=Parameters(1).ParamF(7);
n1=Parameters(1).ParamF(3);                                   %refraction index of incident medium
n2Vec=Parameters(1).ParamVec(2,:);                            %refraction index of transmission medium
thI=Parameters(1).ParamF(2)*pi/180;                           %incident angle (radians)
Spectrum=zeros(2,length(VecWaveLength));                      % preallocate field for the optical response parameters
output_plot = Switches(2).PlotFields;

tic                                                           %start counting computing time
%% Start computation

switch output_plot  %switch between diffraction efficiencies and ellipsometric parameters
 case 'DEF'  %case 0th order diffraction efficiency
    
    % Set permittivity and permeability for the case of TM polarization
    if strcmp(Switches(1).Polar,'TM')   %TM polarization
        mu=1;                           %relative permeability
        eps1=n1*n1/(mu^2);              %relative permittivity of incident medium
        eps2=n2Vec.*n2Vec./(mu^2);      %relative permittivity of transmission medium
    end
    
    % Set permittivity and permeability for the case of TM polarization
    if strcmp(Switches(1).Polar,'TE') %TE polarization
        eps1=1.*ones(1,length(VecWaveLength));                %relative permittivity of incident medium
        eps2=1.*ones(1,length(VecWaveLength));                %relative permittivity of transmitted medium
        mu1=n1.*ones(1,length(VecWaveLength));                %relative permeability of incident medium
        mu2=n2.*ones(1,length(VecWaveLength));                %relative permeability of transmitted medium
    end
    
    for ii=1:length(VecWaveLength); %start computing diffraction efficiencies for given wavelengths
    Lam=VecWaveLength(ii);
    [RVec, real_Ray1_idx, real_Ray2_idx,m1, m2, B1, B2, b0]=Make_Computation( Lam, eps1, eps2(ii), 0, n2Vec(ii) );   
%% Print Efficiencies
        nDim=m2-m1+1;
        etaR=zeros(1,length(real_Ray1_idx));
        etaT=zeros(1,length(real_Ray2_idx));
        for k=min(real_Ray1_idx):max(real_Ray1_idx)
            etaR(k-min(real_Ray1_idx)+1)=(sqrt(B1(k-m1+1))/b0)*(abs(RVec(k-min(real_Ray1_idx)+1)))^2;
        end
        
        for k=min(real_Ray2_idx):max(real_Ray2_idx)
            etaT(k-min(real_Ray2_idx)+1)=(eps1/eps2(ii))*(sqrt(B2(k-m1+1))/b0)*(abs(RVec(k-min(real_Ray2_idx)+1+nDim)))^2;
        end
        
        RAmp=(real_Ray1_idx==0);
        TAmp=(real_Ray1_idx==0);
        Spectrum(1,ii)=etaR(RAmp);
        
        if imag(n2Vec(ii))==0
            Spectrum(2,ii)=etaT(TAmp);
        end
    end
    
%% Plot graphs
    if(imag(n2Vec(ii))~=0)
        figure(2)
        plot(VecWaveLength,Spectrum(1,:))
        title(strcat('Incident angle \vartheta= ',num2str(thI/pi*180),'°'))
        xlabel('wavelength [m]')
        ylabel('0th order reflectance')
        hold on

    else
        figure(2)
        subplot(1,2,1)
        plot(VecWaveLength,Spectrum(1,:),'r')
        title(strcat('Incident angle \vartheta= ',num2str(thI/pi*180),'°'))
        xlabel('wavelength [m]')
        ylabel('0th order reflectance')

        subplot(1,2,2)
        plot(VecWaveLength,Spectrum(2,:),'r')
        title(strcat('Incident angle \vartheta= ',num2str(thI/pi*180),'°'))
        xlabel('wavelength [m]')
        ylabel('0th order transmittance')
    end   
    
 case 'EMP'

   for jj=1:2
        %% TM Polarization
        if jj==1
            mu=1;                   %relative permeability
            eps1=n1*n1/(mu^2);      %relative permittivity of incident medium
            eps2=n2Vec.*n2Vec./(mu^2);      %relative permittivity of transmission medium
        end
        %% TE Polarization
        if jj==2
            eps1=1;                %relative permittivity of incident medium
            eps2=1.*ones(1,length(VecWaveLength));                %relative permittivity of transmitted medium
            mu1=n1.*ones(1,length(VecWaveLength));                %relative permeability of incident medium
            mu2=n2Vec;                                            %relative permeability of transmitted medium
        end
        %% Main computation
        for ii=1:length(VecWaveLength); %start computing diffraction efficiencies for given wavelengths
            Lam=VecWaveLength(ii);            
            [RVec, real_Ray1_idx, real_Ray2_idx,m1, m2, B1, B2, b0]=Make_Computation( Lam, eps1, eps2(ii), 0, n2Vec(ii) );
            Spectrum(jj,ii)=RVec(-min(real_Ray1_idx)+1);       
        end
   end
   
    %% Handle ellipsometric parameters
    
    rho=Spectrum(1,:)./Spectrum(2,:);
	psi=atan(abs(rho))*180/pi;
    delta=angle(rho)*180/pi;
    Ellipso=Parameters(1).Ellipso;
    
    % Plot ellipsometric parameters
    figure(2)
    subplot(1,2,1)
    plot(VecWaveLength,psi,'r',Ellipso(:,1),Ellipso(:,2),'*k')
    title(strcat('Incident angle \vartheta= ',num2str(thI/pi*180),'°'))
    xlabel('wavelength [m]')
    ylabel('\psi [deg]')

    subplot(1,2,2)
    plot(VecWaveLength,delta,'r',Ellipso(:,1),-Ellipso(:,3),'*k')
    title(strcat('Indicend angle \vartheta= ',num2str(thI/pi*180),'°'))
    xlabel('wavelength [m]')
    ylabel('\Delta [deg]')
end


%% Plot fields if required
if strcmp(Switches(1).PlotFields,'YES') 
    Lam = WaveLengthToPlot;
    [c n2_ind] = min(abs(Lam-VecWaveLength)); %find the closest refraction index
    [RVec, real_Ray1_idx, real_Ray2_idx,m1, m2, B1, B2, b0]=Make_Computation( Lam, eps1, eps2(ii), 1, n2Vec(n2_ind) ); 
end
%clearvars -except etaR etaT Lam WaveLengthToPlot VecWaveLength Parameters d h thI n1 n2 K mu0 eps0 nTr tol alfa0 mu eps1 eps2 mu1 mu2 nDim real_Ray1_idx i RefVec


clear real_Ray1_idx
toc
