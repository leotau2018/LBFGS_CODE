%%���ÿ�������Ϣ���������࣬ͬʱ���ݷ������õ��������򡣴˴����ǵ�����F_lu��ȴ����߽�ܽ��������
function [ds_e,ds_lu]=getD_byCauchy( S, Y, n, start_index, gamma, M, k_e, k_lu, bb_lu, zc_lu, zb_l,w, z0_lu,g)
em=min(w,1e-8);
k=size(S,1);
%ds=zeros(k,1);
%ds_lu=zeros(k_lu,1);
%dif_lu=zc_lu-z0_lu;%%����C�еĻ���Լ���������ø��ݶȷ��򣬾���Ϊx_c-x_k,����F�еķǻ���Լ������������ţ�ٷ���
ds=[zeros(k_e,1);zc_lu-z0_lu];

C1=find(abs(zc_lu-zb_l<em));   %C1��Ӧ�ļ�����Ϊ�ܿ����߽磬���ܲ�ȡ��ţ�ٷ�����˲�ȡ���ݶȷ��򱣳��½���������Ҳ�Ǹ��ݶȷ���
C=union(bb_lu(1:end),C1);            %bb_luΪ��������Ϣɸѡ�����Ļ���Լ������C1Ϊ��x_k���ǳ������߽��Լ��������

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

Wkb=[gamma*Y,S];                                                      %LBFGS��ر�������
SY=S'*Y;
Dk=diag(diag(SY));
Rk=triu(SY);Rk=inv(Rk);
Mkb=[zeros(n),-Rk;-Rk',Rk'*(Dk+gamma*(Y'*Y))*Rk];


NNg=zeros(k,1);
NNg(F)=-g(F);
tmp=Wkb*(Mkb*(Wkb'*NNg));
tmp=tmp+gamma*NNg;
ds(F)=tmp(F);        %ds�Ǽ���û��Լ��ʱʹ����ţ�ٷ�����������

%����Ϊ�˸�ʽͳһ����������
ds_e=ds(1:k_e);           %length of ds_e=k_e
ds_lu=ds(k_e+1:end);        %length of ds_lu=k_lu,�˴�ΪԼ�����ӵķǻ���Լ�����ֲ�����ţ�ٷ���
%ds_lu(C)=dif_lu(C);       %������Ҳ�Ǹ��ݶȷ��򡣼�����Լ����C�ķ��������ſ�ϧ���߲���Ϊ1ʱ���������
end