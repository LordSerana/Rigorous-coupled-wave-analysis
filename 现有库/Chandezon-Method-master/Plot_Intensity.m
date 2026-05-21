function [ ] = Plot_Intensity( d,k0,h,alfa0,b0,a_fun,RVec,real_Ray1_idx, SB1, A, real_Ray2_idx, real_eig1p, imag_Vec1p, imag_eig1p, SB2, real_eig2n, imag_Vec2n,imag_eig2n, m1, m2, save_plot, Lam)
%function Plot_Intensity - generates intensity plot of the electric (TE case) or
%magnetic (TM case) field. This plot can be saved to a file. Input
%parameters are grating and incident light parameters and computed fields
%


    disp('Plotting intensity distribution');
    %% Plot fields
    nPoints=200;                            % number of nodes
    xx=linspace(0,d*k0,nPoints);            % generates x-grid
    yy=linspace(-0.7*h*k0,h*k0,nPoints);    % generates y-grid
    [X,Y]=meshgrid(xx,yy);                  % generates [X,Y] grid
    clear yy
    wavelength=Lam;
    FIn=exp(1i*alfa0/k0*X-1i*b0*Y).*(a_fun(X)<=Y);  % Values of the input field on the grid
    FRPlus=zeros(nPoints);          %preallocate fields for positive propagation orders in superstrate medium
    FRNeg=zeros(nPoints);           %preallocate fields for negative propagation orders in substrate medium
    FRPlusIm=zeros(nPoints);        %preallocate fields for positive evanescent orders in superstrate medium
    FRNegIm=zeros(nPoints);         %preallocate fields for negative evanescent orders in substrate medium
    
    % computes values of reflected propagating orders on the grid 
    for mm=1:length(real_eig1p)
        FRPlus=(FRPlus + RVec(mm).*exp(1i*A(mm+min(real_Ray1_idx)-m1).*X + 1i*SB1(mm+min(real_Ray1_idx)-m1).*Y)).*(a_fun(X)<=Y);
    end
     
    % computes values of reflected evanescent orders on the grid
    for mm=m1:m2
        for kk=1:length(imag_eig1p)
            FRPlusIm=FRPlusIm + (exp(1i*A(mm-m1+1).*X).*((RVec(kk+length(real_eig1p))*imag_Vec1p(mm-m1+1,kk)).*exp(1i*imag_eig1p(kk).*(Y-a_fun(X))))).*(a_fun(X)<=Y);
        end
    end
    
    disp('Keep waiting, we are working on it')  % computation can be quite log
    
    % computes values of transmitted propagation orders on the grid
    for mm=1:length(real_eig2n)
       FRNeg=FRNeg + (RVec(mm+length(real_eig1p)+length(imag_eig1p)).*exp(1i*A(mm+min(real_Ray2_idx)-m1).*X + 1i*SB2(mm+min(real_Ray2_idx)-m1).*Y)).*(a_fun(X)>Y);
    end

    % computes values of transmitted evanescent orders on the grid
    for mm=m1:m2
        for kk=1:length(imag_eig2n)
           FRNegIm=FRNegIm + (exp(1i*A(mm-m1+1).*X).*((RVec(kk+length(real_eig1p)+length(imag_eig1p)+length(real_eig2n))*imag_Vec2n(mm-m1+1,kk)).*exp(1i*imag_eig2n(kk).*(Y-a_fun(X))))).*(a_fun(X)>Y);
        end
    end
    
    disp('It is almost done')
    
    Z=double(abs(FIn+FRPlus+FRPlusIm+FRNeg+FRNegIm));   % Computes the whole intenisty on the grid
    %Z=double(real(FIn+FRPlus+FRPlusIm+FRNeg+FRNegIm)); % Computes the values(!) of the electric/magnetic field on the grid.
    
    %% Start plot :-)
    figure('Name','Distribution of intensity')
    hold on
    pcolor(X,Y,Z);
    shading flat;
    colorbar;
    colormap jet;
    axis equal;
    fy=a_fun(xx); %plot the grating profile
    IntensityPlot=plot(xx,fy,'w-','LineWidth',1.5); %plot the intensity
    
    % save to file if required
    if strcmp(save_plot,'YES')
       file_name=strcat('Intensitydistribution',num2str(wavelength),'.png');
       saveas(IntensityPlot, file_name); %Save plot to png file
    end
    hold off
    
    

end

