function [ds_e,ds_lu]=getDs_Free( S, Y, n, start_index, gamma, M, k_e, k_lu, F_lu, bb_lu, zc_lu, zb_l,gc)
%根据柯西点得到的积极约束集和非积极约束集计算x_k(不是柯西点)处的搜索方向
%generate the search direction at pre-cauchy point(x_k，not x_c) as to reach L-bound useing
%lbfgs method
C3=bb_lu;%BC2=F_LU;
if n==0
    ds_e=-gc(1:k_e);
    ds_lu=-gc(k_e+1 : k_e+k_lu);
else
    k=size(S,1);
    ds=zeros(k,1);
    ds_lu=zeros(k_lu,1);               %search direction 属于C1的分量的搜索方向为零
    dif_lu=zc_lu-zb_l;
    ds_lu(C3)=-dif_lu(C3);             %自由变量部分的搜索方向更新，即一步拉到了l bound的方向（步长为1）
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
    BC2=F_lu;
    Be=(1:1:k_e)';
    BeBC2=[Be;k_e+BC2];
    NNg=zeros(k,1);
    NNg(BeBC2)=-gc(BeBC2);
    tmp=Wkb*(Mkb*(Wkb'*NNg));
    tmp=tmp+gamma*NNg;
    ds(BeBC2)=tmp(BeBC2);

    %以下为了格式统一而做调整。
    ds_e=ds(Be);          %length of ds_e=k_e
    ds_lu(BC2)=ds(BC2+k_e);%length of ds_l=k_l
end
end

