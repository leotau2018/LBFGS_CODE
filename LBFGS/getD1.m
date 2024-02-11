%已放弃代码18/04
function [g_ze,g_zl,g_zu,lam]=getD1(zp_e, zp_l, zp_u,d_l c, gamma, S, Y ) %getCauchyg(...)
S(:,(sum(abs(S))<esp))=[];                                     %将整列为零的列删去
Y(:,(sum(abs(Y))<esp))=[];
Wk=[Y,S];                                                      %LBFGS相关变量计算
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
%%%投影函数
function x=proje(x,l,u)
x((x-l)<-eps)=l((x-l)<eps);     %向l bond投影。
x((x-u)> eps)=u((x-u)> eps);     %向u bond 投影。
end
