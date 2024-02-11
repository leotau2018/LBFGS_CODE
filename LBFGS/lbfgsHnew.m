function [ds_e,ds_l]=lbfgsHnew( S, Y, n, start_index, gamma, M, k_l, C1, C2, C3, B,zb_l,gp)
%renew the appropration of hessian matrix.
ds_l=zeros(k_l,1);               %search direction ����C1�ķ�������������Ϊ��
dif_l=zc_l-zb_l;
ds_l(C3)=-dif_l(C3);             %���ɱ������ֵ�����������£���һ��������l bond�ķ��򣨲���Ϊ1��

if n==0
    H = speye(S);
return;
end
if n==M && start_index>1
    ind=[start_index:M, 1:start_index-1];
    S=S(:,ind);
    Y=Y(:,ind);
else
    ind=1:n;
    S=S(:,ind);
    Y=Y(:,ind);
end
k=length(S,1);
Wkb=[gamma*Y,S];                                                      %LBFGS��ر�������
SY=S'*Y;
Dk=diag(diag(SY));
Rk=triu(SY);Rk=Rk^-1;
Mkb=[zeros(M),-Rk;-Rk',Rk'*(Dk+1/gamma*Y'*Y)*Rk];
BC2=[B,C2];
Be=1:1:k_e;
BeBC2=[Be,BC2];
LocBeBC2  =zeros(k,1);
LocBeBC2(BeBC2)=1;
ds_BeBC2=zeros(k,1);
ds_BeBC2=LocBeBC2.*gp;      %cost k opreations
Wds  =Wkb'*ds_BeBC2;       %cost 2*M*kopreations
MWds =Mkb*Wds;             %cost 2*M*kopreations
WMWds=Wkb*MWds;            %cost 2*M*kopreations
ds_BeBC2=ds_BeBC2+WMWds;   %��ʵ�������ֵ�������һ��Ϊ�˸�ʽͳһ��
ds_e=ds_BeBC2(Be);
ds_l(BC2)=ds_BeBC2(BC2);
end
   
