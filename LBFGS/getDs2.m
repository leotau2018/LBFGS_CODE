%getDs2,用来计算第二种情况下的搜索方向。
function [ds_e,ds_lu]=getDs2( S, Y, n, start_index, gamma, M, k_e, k_lu, CC1, CC2, B, z_lu, zb_l,g)
%generate the alter search direction-d2 at current step kth point as to reach L-bound useing
%lbfgs method
if n==0
    ds_e=-g(1:k_e);
    ds_lu=-g(k_e+1 : k_e+k_lu);
else
    k=size(S,1);
    ds=zeros(k,1);
    ds_lu=zeros(k_lu,1);                 %search direction 属于C1的分量的搜索方向为零,其实无所谓，第二种分类将已在边界上的分量归为CC1中
    dif_lu=z_lu-zb_l;
    ds_lu(CC1)=-dif_lu(CC1);             %自由变量部分(CC1)的搜索方向更新，即一步拉到了l bound的方向（步长为1）
    ds_lu(CC2)=-g(CC2+k_e);                    %属于CC2的分量采用负梯度方向。
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
    BeB=[Be;k_e+B];
    NNg=zeros(k,1);
    NNg(BeB)=-g(BeB);
    tmp=Wkb*(Mkb*(Wkb'*NNg));
    tmp=tmp+gamma*NNg;
    ds(BeB)=tmp(BeB);      %%%ds是假设没有约束时使用拟牛顿法的搜索方向；其实此处可以省取tmp

    %以下为了格式统一而做调整。
    ds_e=ds(Be);          %length of ds_e=k_e
    ds_lu(B)=ds(B+k_e);%length of ds_l=k_l
end
end

