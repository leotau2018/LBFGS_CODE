%% 按照d2的标准将所有分量分成四个指标集，
%C1为有效约束集,搜索方向为零；C2为临近边界的指标集，搜索方向为负梯度方向；C3为临近边界的积极约束，搜索方向为特殊的抵达边界方向；
function [CC1,CC2,B]=classify2(zc_lu, gp_zlu, zb_l, w)
em=min(w,1e-6);
CC1=find(abs(zc_lu-zb_l)<=em  & gp_zlu>-em);
CC2=find(abs(zc_lu-zb_l)<=em  & gp_zlu<=-em);
% C3=find(abs(zc_lu-zb_l)>0    & abs(zc_lu-zb_l)<=em  & gp_zlu>=0);
% C=[CC1;CC2;C3];
C=[CC1;CC2];
B =(1:1:length(zc_lu))';
B(C)=[];
end