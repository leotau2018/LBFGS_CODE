%�ѷ�������18/04
function [g_ze,g_zl,g_zu,lam]=getD1(zp_e, zp_l, zp_u,d_l c, gamma, S, Y ) %getCauchyg(...)
S(:,(sum(abs(S))<esp))=[];                                     %������Ϊ�����ɾȥ
Y(:,(sum(abs(Y))<esp))=[];
Wk=[Y,S];                                                      %LBFGS��ر�������
SY=S'*Y;
Dk=diag(diag(SY));
Lk=tril(SY,-1);
Mk=[-Dk,Lk';Lk,1/gamma*S'*S]^-1;

B0=speye(length(d_l));
Zk=B0(:,F_l);
gc_zl=()

%compute the
end
%}

nrmw=norm(d_l(F_l), inf);
if nrmw< tol
    out.msg='converge';
    break;
end
%%%ͶӰ����
function x=proje(x,l,u)
x((x-l)<-eps)=l((x-l)<eps);     %��l bondͶӰ��
x((x-u)> eps)=u((x-u)> eps);     %��u bond ͶӰ��
end
