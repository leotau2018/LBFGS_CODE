clear all
  
tau = .0e-8;


%% initial of the problem 
%% Case I 
% %%%%%% RiskMetrics Data (n =387)
% load x.mat
% G = subtract(x);
% G  = (G +G')/2;
% [n, n_c]=size(G);
%%%%%%%%%%%%%%%%%%%%%%%%%%
 n =1000;

%% Case II
% % % % %%%%%%%%%%%%%%%%%%%%%
% n = 500;
% x = 10.^[-4:4/(n-1):0];
% G = gallery('randcorr',n*x/sum(x));
% %%%%%%%%%%%%%%%%%%%%%%%%%%

% for i=1:n
%     G(i,i) =1;
% end




%% Case I
E = 2.0*rand(n,n) - ones(n,n);
E = triu(E) + triu(E,1)';
E=E'*E;
alpha = .1;
%G = (1-alpha)*G+ alpha*E;
 
G = E;
for i=1:n
    G(i,i) =1;
end




%%%%%%%%%%%%%%%%%%%%%%%%% Constraints                               
lh =  1;   % number of fixed off diagonal elements in each row
ll =  0;    % number of off diagonal elements of lower bounds in each row
lu =  0;    % number of off diagonal elements of upper bounds in each row
ll = min(ll,n-1); 
lu = min(lu,n-1);
lh = min(lh,n-1);


%% I_e,J_e
%%%% for fixed  diagonal entries
I_d = [1:1:n]';
J_d = I_d;
%%%% for fixed off-diagonal entries
I_h = [];
J_h = [];
for i = 1:n-lh
    r = rand(n-i,1);
    [r,ind] = sort(r);
    I_h = [I_h; i*ones(lh,1)];
    J_h = [J_h; i+ind(n-i-lh+1:n-i)];
end
for i = ((n-lh)+1):(n-1)
    I_h = [I_h; i*ones(n-i,1)];
    J_h = [J_h;[(i+1):n]'];
end
k_h = length(I_h);

I_e = [I_d;I_h];
J_e = [J_d;J_h];
k_e = length(I_e);


%%  I_l,J_l 
%%%  entries with lower bounds
I_l = [];
J_l = [];
for i = 1:n-ll
    r = rand(n-i,1);
    [r,ind] = sort(r);
    I_l = [I_l; i*ones(ll,1)];
    J_l = [J_l; i+ind(n-i-ll+1:n-i)];
end
for i = ((n-ll)+1):(n-1)
    I_l = [I_l; i*ones(n-i,1)];
    J_l = [J_l;[(i+1):n]'];
end
k_l = length(I_l);


%%  I_u,J_u      
%%%%%  entries with upper bounds
I_u = [];
J_u = [];
for i = 1:n-lu
    r = rand(n-i,1);
    [r,ind] = sort(r);
    I_u = [I_u; i*ones(lu,1)];
    J_u = [J_u; i+ind(n-i-lu+1:n-i)];
end
for i = ((n-lu)+1):(n-1)
    I_u = [I_u; i*ones(n-i,1)];
    J_u = [J_u;[(i+1):n]'];
end
k_u = length(I_u) ;





%% to generate the bound e,l & u
%%%%%%% e
rhs    = ones(n,1);  % diagonal elements
alpha0 = 1;
rhs    = alpha0*rhs + (1-alpha0)*rand(n,1);
h      = zeros(k_h,1);
e      = [rhs;h];
%%%%%%% l
l = -0.10*ones( k_l,1);
%l = 0.50 * (2*rand(k_l,1)-ones(k_l,1));
%l = 1.0 * (rand(k_l,1) - ones(k_l,1));

%%%%%%% u
u = 0.10*ones(k_u,1);
%u = 1.0*(rand(k_l,1) - ones(k_l,1));


max_l = max(l);
min_l = min(l);
max_u = max(u);
min_u = min(u);

 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
Ind = find(h==0);
for i = 1:length(Ind)
    G( I_h(Ind(i)),J_h(Ind(i)) ) = 0;
    G( J_h(Ind(i)),I_h(Ind(i)) ) = 0;
end
G = G - diag(diag(G)) + eye(n);  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 ConstrA.e = e; ConstrA.Ie = I_e; ConstrA.Je = J_e;
 ConstrA.l = l; ConstrA.Il = I_l; ConstrA.Jl = J_l;
 ConstrA.u = u; ConstrA.Iu = I_u; ConstrA.Ju = J_u;
 
 global thetapar
thetapar.G=G;
thetapar.e=e;
thetapar.I_e=I_e;
thetapar.J_e=J_e;
thetapar.I_l=I_l;
thetapar.J_l=J_l;
thetapar.I_u=I_u;
thetapar.J_u=J_u;
thetapar.k_e=k_e;
thetapar.k_l=k_l;
thetapar.k_u=k_u;
thetapar.l=l;
thetapar.u=u;
thetapar.n=n;
 
 
 

OPTIONS.tau = tau; 
OPTIONS.tol = 1e-5;

[X,z,info] = CaliMat1Mex(G,ConstrA,OPTIONS);
%{ 
k=k_e+k_l+k_u;
z0=zeros(k,1);
lb=-inf(k,1);
ub=-lb;
lb(k_e+1:k)=0;
tic;zx = lbfgsb(z0,lb,ub,'thetafun2','thetafung2',...
           [],'genericcallback','maxiter',800,'m',4,'factr',1e-12,...
           'pgtol',1e-5);
       toc;
z1.e=zx(1:k_e);
z1.l=zx(k_e+1:k_e+k_l);
z1.u=zx(k_e+k_l+1:k);
%}
%{%
[X1,z1,info] = CaliMatLbfgs(G,ConstrA,OPTIONS);
nrmz=norm([z.e;z.l;z.u]-[z1.e;z1.l;z1.u])
[theta,g_ze,g_zl,g_zu,eig_time] = thetafun(G,e,z.e,I_e,J_e,l,z.l,I_l,J_l,u,z.u,I_u,J_u,n);
 nrmG = norm([g_ze;g_zl;g_zu], 'fro');
 [theta1,g_ze,g_zl,g_zu,eig_time] = thetafun(G,e,z1.e,I_e,J_e,l,z1.l,I_l,J_l,u,z1.u,I_u,J_u,n);
 nrmG1 = norm([g_ze;g_zl;g_zu], 'fro');
 diftheta=theta-theta1
 difnrm=nrmG-nrmG1
%}
disp('=====end========end==========end===========end=============end==========end==========')
 
 
 
   
 
 
 