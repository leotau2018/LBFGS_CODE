function [ds_e,ds_lu]=getDs( S, Y, n, start_index, gamma, M, k_e, k_lu, C1, C2, C3, B, zc_lu, zb_l,gc)
%generate the search direction at cauchy point as to reach L-bound useing
%lbfgs method
if n==0
    ds_e=-gc(1:k_e);
    ds_lu=-gc(k_e+1 : k_e+k_lu);
else
    k=size(S,1);
    ds=zeros(k,1);
    ds_lu=zeros(k_lu,1);               %search direction ����C1�ķ�������������Ϊ��
    dif_lu=zc_lu-zb_l;
    ds_lu(C3)=-dif_lu(C3);             %���ɱ������ֵ�����������£���һ��������l bound�ķ��򣨲���Ϊ1��
    if n==M && start_index>1
        ind=[start_index:M, 1:start_index-1];
        S=S(:,ind);
        Y=Y(:,ind);
    else
        ind=1:n;
        S=S(:,ind);
        Y=Y(:,ind);
    end
    
    Wkb=[gamma*Y,S];                                                      %LBFGS��ر�������
    SY=S'*Y;
    Dk=diag(diag(SY));
    Rk=triu(SY);Rk=inv(Rk);
    Mkb=[zeros(n),-Rk;-Rk',Rk'*(Dk+gamma*(Y'*Y))*Rk];
    BC2=[B;C2];
    Be=(1:1:k_e)';
    BeBC2=[Be;k_e+BC2];
    NNg=zeros(k,1);
    NNg(BeBC2)=-gc(BeBC2);
    tmp=Wkb*(Mkb*(Wkb'*NNg));
    tmp=tmp+gamma*NNg;
    ds(BeBC2)=tmp(BeBC2);

    %����Ϊ�˸�ʽͳһ����������
    ds_e=ds(Be);          %length of ds_e=k_e
    ds_lu(BC2)=ds(BC2+k_e);%length of ds_l=k_l
end
end

