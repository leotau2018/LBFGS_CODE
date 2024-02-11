%%% dual function theta(z) and its gradient
function  [theta,g_ze,g_zl,g_zu,eig_time] = ...
   thetafun(G,e,z_e,I_e,J_e,l,z_l,I_l,J_l,u,z_u,I_u,J_u,n)

k_e = length(e);
k_l = length(l);
k_u = length(u);
k   = k_e + k_l + k_u;

g_ze = zeros(k_e,1);
g_zl = zeros(k_l,1);
g_zu = zeros(k_u,1);

X = zeros(n,n);
for i=1:k_e
  X(I_e(i), J_e(i)) = z_e(i);
end
for i=1:k_l
  X(I_l(i), J_l(i)) = z_l(i) + X(I_l(i), J_l(i));
end
for i=1:k_u
  X(I_u(i), J_u(i)) = -z_u(i) + X(I_u(i), J_u(i));  %%% upper bound
end
X = 0.5*(X + X');
X = G + X;
X = (X + X')/2;
t1         = clock;
[P,lambda] = MYmexeig(X);
eig_time   = etime(clock,t1);
Xplus=P*diag(max(0,lambda))*P';
bz=e'*z_e+l'*z_l-u'*z_u;
theta=norm(Xplus,'fro')^2/2-bz;

perp=0;
for i=1:k_e
  g_ze(i)=Xplus(I_e(i), J_e(i));
end
g_ze=g_ze-e+perp*randn(k_e,1);
for i=1:k_l
  g_zl(i)=Xplus(I_l(i), J_l(i));
end
g_zl=g_zl-l+perp*randn(k_l,1);
for i=1:k_u
  g_zu(i)=-Xplus(I_u(i), J_u(i));  %%% upper bound
end
g_zu=g_zu+u+perp*randn(k_u,1);

return
