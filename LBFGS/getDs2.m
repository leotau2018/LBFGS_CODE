%getDs2,��������ڶ�������µ���������
function [ds_e,ds_lu]=getDs2( S, Y, n, start_index, gamma, M, k_e, k_lu, CC1, CC2, B, z_lu, zb_l,g)
%generate the alter search direction-d2 at current step kth point as to reach L-bound useing
%lbfgs method
if n==0
    ds_e=-g(1:k_e);
    ds_lu=-g(k_e+1 : k_e+k_lu);
else
    k=size(S,1);
    ds=zeros(k,1);
    ds_lu=zeros(k_lu,1);                 %search direction ����C1�ķ�������������Ϊ��,��ʵ����ν���ڶ��ַ��ཫ���ڱ߽��ϵķ�����ΪCC1��
    dif_lu=z_lu-zb_l;
    ds_lu(CC1)=-dif_lu(CC1);             %���ɱ�������(CC1)������������£���һ��������l bound�ķ��򣨲���Ϊ1��
    ds_lu(CC2)=-g(CC2+k_e);                    %����CC2�ķ������ø��ݶȷ���
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
    %BC2=[B;C2];
    Be=(1:1:k_e)';
    BeB=[Be;k_e+B];
    NNg=zeros(k,1);
    NNg(BeB)=-g(BeB);
    tmp=Wkb*(Mkb*(Wkb'*NNg));
    tmp=tmp+gamma*NNg;
    ds(BeB)=tmp(BeB);      %%%ds�Ǽ���û��Լ��ʱʹ����ţ�ٷ�������������ʵ�˴�����ʡȡtmp

    %����Ϊ�˸�ʽͳһ����������
    ds_e=ds(Be);          %length of ds_e=k_e
    ds_lu(B)=ds(B+k_e);%length of ds_l=k_l
end
end

