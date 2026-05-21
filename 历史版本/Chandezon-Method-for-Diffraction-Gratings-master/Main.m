% Matlab implementation of the Chandezon method (C method) given in
% L. Li et al., 1999, title "Rigorous and efficient grating-analysis method made easy for optical engineers"
clear; clc ; close all;

%% polarization 'TE' or 'TM'
Polarization='TE';

%% material properties
n1 = 1;       % refractive index of the first material (ambient)
n2 = 1.4482+7.5367i; % refractive index of the second material
StrucParam = SetConstantsByPolarization(n1, n2, Polarization); % arrangement for TE and TM polarizations

%% angle of incidence -> (0, pi/2), not including borders: 0 - normal incidence, pi/2 - grazing angle
StrucParam.theta = 0;

%% truncation order of the harmonics
StrucParam.N_Tr = 2*12 + 1; % odd number

%% wavelength range in micrometers
wavelength=632.8*1e-9;

%% strucutre period micrometers
StrucParam.dx = 4*1e-6;

%% flag - accuracy rounding 
StrucParam.cut = 1;            % 0 - default, no rounding, 1 - round to zero small elements in the array a_diff and all F and G matrices
StrucParam.accuracy = 1e-9;   % set the threshold for rounding

%% accuracy of imaginary part of k-vector
StrucParam.kVecImagMin = 1e-8;% default, a k-vector is set to be real if abs(Imag(k)) is smaller than this threshold

%% Example 1: profile of the currugation: example ridge-interface with 30 degr slope %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% StrucParam.alpha = 30*pi/180;                                                                                                               %
% StrucParam.a_x = '-tan(StrucParam.alpha)*abs(x-(StrucParam.dx/2))+tan(StrucParam.alpha)*StrucParam.dx/2'; % a(x) - profile                  %
% StrucParam.diff_a_x = 'sign(StrucParam.dx/2 - x)*tan(StrucParam.alpha)';                                  % derivative of the a(x)-profile  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Example 2: profile composed of even Fourier harmonics (four terms) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% StrucParam.An=[0.12 0.01 0.02 0.02]; % Fourier-coefficients
%                                                                                                                                            %
% StrucParam.a_x=['StrucParam.An(1)*cos(2*pi*x*1/StrucParam.dx)+',...                                                                        %
%                 'StrucParam.An(2)*cos(2*pi*x*2/StrucParam.dx)+',...                                                                        %
%                 'StrucParam.An(3)*cos(2*pi*x*3/StrucParam.dx)+',...                                                                        %
%                 'StrucParam.An(4)*cos(2*pi*x*4/StrucParam.dx)'];                                                                           %
%                                                                                                                                            %
% StrucParam.diff_a_x=['-StrucParam.An(1)*(2*pi*1/StrucParam.dx)*sin(2*pi*x*1/StrucParam.dx)',...                                            %
%                      '-StrucParam.An(2)*(2*pi*2/StrucParam.dx)*sin(2*pi*x*2/StrucParam.dx)',...                                            %
%                      '-StrucParam.An(3)*(2*pi*3/StrucParam.dx)*sin(2*pi*x*3/StrucParam.dx)',...                                            % 
%                      '-StrucParam.An(4)*(2*pi*4/StrucParam.dx)*sin(2*pi*x*4/StrucParam.dx)'];                                              % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Example 3:Sinosudal grating
StrucParam.a_x='1*1e-6*(1+sin(2*pi/(4*1e-6)*x))';
StrucParam.diff_a_x='1/2*pi*cos(2*pi/(4*1e-6)*x)';
%% Start calculations
R_tot = zeros(1,length(wavelength));
T_tot = zeros(1,length(wavelength));

for itWavelength = 1:length(wavelength)
    StrucParam.lambda = wavelength(itWavelength);
    timerVal = tic;
    [R_tot(itWavelength), T_tot(itWavelength)] = C_method_compute(StrucParam);
    disp(['Iteration ', num2str(itWavelength),' out of ',num2str(length(wavelength)), ', took ', num2str(toc(round(timerVal)),2),' sec']);
end

PhC = PhysConstCGS;

GG = 2*pi/(StrucParam.dx*1e-4);

% plot(2*pi*nm2Hz(1e3*wavelength)/PhC.c/GG, R_tot,'.-k','Markersize',15); hold on;
% xlabel('$$\omega/cG$$','Interpreter','Latex');hold on;
% ylabel('Total Reflection');hold on;
% plot(2*pi*nm2Hz(1e3*wavelength)/PhC.c/GG, T_tot); hold on;
% plot(2*pi*nm2Hz(1e3*wavelength)/PhC.c/GG, T_tot+R_tot); hold on;