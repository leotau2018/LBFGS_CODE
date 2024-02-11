%%利用柯西点信息将变量分类，同时根据分类结果得到搜索方向。此处考虑到了在F_lu中却距离边界很近的情况。
function [ds_e,ds_lu]=getD_byCauchy( S, Y, n, start_index, gamma, M, k_e, k_lu, bb_lu, zc_lu, zb_l,w, z0_lu,g)
em=min(w,1e-8);
k=size(S,1);
%ds=zeros(k,1);
%ds_lu=zeros(k_lu,1);
%dif_lu=zc_lu-z0_lu;%%对于C中的积极约束集，采用负梯度方向，具体为x_c-x_k,对于F中的非积极约束集，采用拟牛顿方向。
ds=[zeros(k_e,1);zc_lu-z0_lu];

C1=find(abs(zc_lu-zb_l<em));   %C1对应的集合因为很靠近边界，不能采取拟牛顿法，因此采取负梯度方向保持下降。柯西步也是负梯度方向。
C=union(bb_lu(1:end),C1);            %bb_lu为柯西点信息筛选出来的积极约束集，C1为在x_k处非常靠近边界的约束分量。

F=[1:k]';
F(C+k_e)=[];

if n==M && start_index>1
    ind=[start_index:M, 1:start_index-1];
    S=S(:,ind);
    Y=Y(:,ind);
else
    ind=1:n;
    S=S(:,ind);
    Y=Y(:,ind);
end

Wkb=[gamma*Y,S];                                                      %LBFGS相关变量计算
SY=S'*Y;
Dk=diag(diag(SY));
Rk=triu(SY);Rk=inv(Rk);
Mkb=[zeros(n),-Rk;-Rk',Rk'*(Dk+gamma*(Y'*Y))*Rk];


NNg=zeros(k,1);
NNg(F)=-g(F);
tmp=Wkb*(Mkb*(Wkb'*NNg));
tmp=tmp+gamma*NNg;
ds(F)=tmp(F);        %ds是假设没有约束时使用拟牛顿法的搜索方向

%以下为了格式统一而做调整。
ds_e=ds(1:k_e);           %length of ds_e=k_e
ds_lu=ds(k_e+1:end);        %length of ds_lu=k_lu,此处为约束乘子的非积极约束部分采用拟牛顿方向
%ds_lu(C)=dif_lu(C);       %柯西步也是负梯度方向。即对于约束集C的分量，朝着可惜点走步长为1时到达柯西点
end