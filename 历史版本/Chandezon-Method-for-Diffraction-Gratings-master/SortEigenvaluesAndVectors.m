function [eig_p,vect_p,eig_n,vect_n]=SortEigenvaluesAndVectors(eig,vect)
%% sortation of eigenvalues and corresp eigenvectros
real_patrs=real(eig);
imag_parts=imag(eig);
abs_real=abs(real_patrs);
abs_imag=abs(imag_parts);
n=length(eig);
is_real_dominant=(abs_real>100*abs_imag);
is_imag_dominant=(abs_imag>100*abs_real);
is_mixed=(~is_real_dominant)&(~is_imag_dominant);
idxRealAndPositive=false(n,1);
idxRealAndNegative=false(n,1);
idxImagAndPositive=false(n,1);
idxImagAndNegative=false(n,1);
%=====================实数主导
real_dominant_indices=find(is_real_dominant);
for i=1:length(real_dominant_indices)
    idx=real_dominant_indices(i);
    if real_patrs(idx)>0
        idxRealAndPositive(idx)=true;
    else
        idxRealAndNegative(idx)=true;
    end
end
%=====================虚数主导
imag_dominant_indices=find(is_imag_dominant);
for i=1:length(imag_dominant_indices)
    idx=imag_dominant_indices(i);
    if imag_parts(idx)>0
        idxImagAndPositive(idx)=true;
    else
        idxImagAndNegative(idx)=true;
    end
end
%===================混合特征值
mixed_indices=find(is_mixed);
for i=1:length(mixed_indices)
    idx=mixed_indices(i);
    if imag_parts(idx)>0
        idxImagAndPositive(idx)=true;
    else
        idxImagAndNegative(idx)=true;
    end
end
eig_real_p=eig(idxRealAndPositive);
vect_real_p=vect(:,idxRealAndPositive);
eig_real_n=eig(idxRealAndNegative);
vect_real_n=vect(:,idxRealAndNegative);
eig_imag_p=eig(idxImagAndPositive);
vect_imag_p=vect(:,idxImagAndPositive);
eig_imag_n=eig(idxImagAndNegative);
vect_imag_n=vect(:,idxImagAndNegative);
if ~isempty(eig_real_p)
    [~,ind]=sort(real(eig_real_p),'descend');
    eig_real_p=eig_real_p(ind);
    vect_real_p=vect_real_p(:,ind);
end
if ~isempty(eig_imag_p)
    [~,ind]=sort(imag(eig_imag_p),'ascend');
    eig_imag_p=eig_imag_p(ind);
    vect_imag_p=vect_imag_p(:,ind);
end
if ~isempty(eig_real_n)
    [~,ind]=sort(real(eig_real_n),'ascend');
    eig_real_n=eig_real_n(ind);
    vect_real_n=vect_real_n(:,ind);
end
if ~isempty(eig_imag_n)
    [~,ind]=sort(imag(eig_imag_n),'descend');
    eig_imag_n=eig_imag_n(ind);
    vect_imag_n=vect_imag_n(:,ind);
end
eig_p=[eig_real_p;eig_imag_p];
vect_p=[vect_real_p,vect_imag_p];
eig_n=[eig_real_n;eig_imag_n];
vect_n=[vect_real_n,vect_imag_n];
end
% idxRealAndPositive = ((abs(imag(eig)) < accuracyImag) & (real(eig) > 0));
% idxRealAndNegative = ((abs(imag(eig)) < accuracyImag) & (real(eig) <= 0));
% idxImagAndPositive = ((abs(imag(eig)) >= accuracyImag) & (imag(eig) > 0));
% idxImagAndNegative = ((abs(imag(eig)) >= accuracyImag) & (imag(eig) <= 0));
% 
% eig_real_p = eig(idxRealAndPositive).';
% vec_real_p = vect(:,idxRealAndPositive);
% eig_real_m = eig(idxRealAndNegative).';
% vec_real_m = vect(:,idxRealAndNegative);
% eig_comp_p = eig(idxImagAndPositive).';
% vec_comp_p = vect(:,idxImagAndPositive);
% eig_comp_m = eig(idxImagAndNegative).';
% vec_comp_m = vect(:,idxImagAndNegative);
% 
% %% sortation in descending order
% if ~isempty(eig_real_p)
%     [~,ind]=sort(real(eig_real_p),'descend');
%     eig_real_p=eig_real_p(ind);
%     vec_real_p=vec_real_p(:,ind);
% end
% 
% if ~isempty(eig_comp_p)
%     [~,ind]=sort(imag(eig_comp_p),'ascend');
%     eig_comp_p=eig_comp_p(ind);
%     vec_comp_p=vec_comp_p(:,ind);
% end
% 
% if ~isempty(eig_real_m)
%     [~,ind]=sort(real(eig_real_m),'ascend');
%     eig_real_m=eig_real_m(ind);
%     vec_real_m=vec_real_m(:,ind);
% end
% 
% if ~isempty(eig_comp_m)
%     [~,ind]=sort(imag(eig_comp_m),'descend');
%     eig_comp_m=eig_comp_m(ind);
%     vec_comp_m=vec_comp_m(:,ind);
% end
% 
% eig_p = [eig_real_p, eig_comp_p];
% vect_p = [vec_real_p, vec_comp_p];
% eig_m = [eig_real_m, eig_comp_m];
% vect_m = [vec_real_m, vec_comp_m];