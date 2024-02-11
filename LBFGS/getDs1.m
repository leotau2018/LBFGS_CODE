%getDs1,用来计算第一种情况下的搜索方向。
function [ds_e,ds_lu]=getDs1( S, Y, n, start_index, gamma, M, k_e, k_lu, C1, C2, C3, B, z_lu, zb_l,g)
%generate the search direction at cauchy point as to reach L-bound useing
%lbfgs method
if n==0
    ds_e=-g(1:k_e);
    ds_lu=-g(k_e+1 : k_e+k_lu);
else
    k=size(S,1);
    ds=zeros(k,1);
    ds_lu=zeros(k_lu,1);               %search direction 属于C1的分量的搜索方向为零
    dif_lu=z_lu-zb_l;
    ds_lu(C3)=-dif_lu(C3);             %自由变量部分的搜索方向更新，即一步拉到了l bound的方向（步长为1）
    ds_lu(C2)=-g(C2+k_e);                  %属于C2的分量采用负梯度方向。
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
    %BC2=[B;C2];
    Be=(1:1:k_e)';
    BeB=[Be;k_e+B];%非积极约束
    NNg=zeros(k,1);
    NNg(BeB)=-g(BeB);
    tmp=Wkb*(Mkb*(Wkb'*NNg));
    tmp=tmp+gamma*NNg;
    ds(BeB)=tmp(BeB);

    %以下为了格式统一而做调整。
    ds_e=ds(Be);          %length of ds_e=k_e
    ds_lu(B)=ds(B+k_e);%length of ds_l=k_l
end
end

