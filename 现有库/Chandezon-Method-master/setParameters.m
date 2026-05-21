%% Set parameters of the system
function [Switches, Parameters]=setParameters()

%% Generate the vector of wavelengths and refraction indices from given values or load them from file
%WaveLengthMin=350e-9; %lower bound on wavelength
%WaveLengthMax=1200e-9; %upper bound on wavelength
%NumberOfWaveLength=70; %number of wavelenght
%LamVec = linspace(WaveLengthMin,WaveLengthMax,NumberOfWaveLength);

%n2=(2.18+4.05i).*ones(1,length(LamVec)); %refraction index in transmission media
%n2=1.5.*ones(1,length(LamVec)); %refraction index in transmission media
%n2=1+5i.*ones(1,length(LamVec)); %refraction index in transmission media


load('Al.mat'); %load the parameters from the file

%% Shape of the grating
d=4e-6; %period of the grating
h=1.1547e-6; %depth of the grating

% Asymetrical profile
%a_fun=@(x) (0.1*h)*cos((2*pi/d).*x) + (0.02*h)*cos((4*pi/d).*x - 5*pi/9);
%a_diff_fun=@(x) -(0.1*h*2*pi/d).*sin((2*pi/d).*x) - (0.02*h*4*pi/d).*(sin((4*pi/d).*x - 5*pi/9));

% Zero profile
%a_fun=@(x) 0.*x;
%a_diff_fun=@(x) 0.*x;

%Symmetrical profile
%a_fun=@(x) (h/2)*cos((2*pi/d).*x);
%a_diff_fun=@(x) -((h/2)*2*pi/d).*sin((2*pi/d).*x);

%Quadratic sinusiodal profile
a_fun=@(x) (x*tan(30*pi/180)).*(x<=d/2)+(h-(x-d/2)*tan(30*pi/180)).*(x>=d/2);
a_diff_fun=@(x) (tan(30*pi/180)).*(x<=d/2)+(-tan(30*pi/180)).*(x>=d/2);

Profile={a_fun, a_diff_fun};


%% Refraction index of incident media and incident angle
n1=1; %set refraction index of incident media
ThetaInc=1e-4*pi/180; %set incident angle (degrees)


%% Plot the fields
PlotIntensity={'NO'};  % plot intensity for given wavelength YES/NO
SaveFieldsPlots={'NO'};  %save plot of intensity YES/NO
WaveLengthToPlot=400e-9; %give wavelength for which the intensity distribution will be plotted

%% System parameters
numTr=15; %Truncation number
tol=10e-10; %error tolerance

output_plot={'DEF'}; % EMP - ellipsimetric parameters, 'DEF' - diffraction efficiencies;
Polarization={'TE'}; %set polarization - only for DEF case, in EMP case will be ignored

%% Load ellipsometric parameters from the experiment
load('ni_ellipso');
%Ellipso=zeros(2,3); % if experimental data are not available

%% Generates a structure from the parameters
ParameterField=[d ThetaInc n1  numTr tol h WaveLengthToPlot];
ParameterVec=[LamVec; n2];
field1='ParamF';
field2='ParamVec';
field3='Prof';
field4='Polar';
field5='PlotFields';
field6='Ellipso';
FieldsPlot=[PlotIntensity output_plot SaveFieldsPlots];
Parameters=struct(field1,ParameterField,field2,ParameterVec,field3,Profile,field6,Ellipso);
Switches=struct(field4,Polarization,field5,FieldsPlot);
