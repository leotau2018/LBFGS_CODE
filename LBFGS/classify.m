%% 按照d1的标准将所有分量分成四个指标集，
%C1为有效约束集,搜索方向为零；C2为临近边界的指标集，搜索方向为负梯度方向；C3为临近边界的积极约束，搜索方向为特殊的抵达边界方向；
function [C1,C2,C3,B]=classify(zc_lu, gp_zlu, zb_l, w)
em=min(w,1e-6);
C1=find(abs(zc_lu-zb_l)<eps  & gp_zlu>-eps);
C2=find(abs(zc_lu-zb_l)<=em  & gp_zlu<0);
C3=find(abs(zc_lu-zb_l)>0    & abs(zc_lu-zb_l)<=em  & gp_zlu>=0);
C=[C1;C2;C3];
B =(1:1:length(zc_lu))';
B(C)=[];
end