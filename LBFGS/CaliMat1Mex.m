function [X,z,info,k1,f_eval,time_used] = CaliMat1Mex(G,ConstrA,OPTIONS,z0)
%%%%%%%%%%%%% This code is designed to solve %%%%%%%%%%%%%%%%%%%%%
%%       min    0.5*<X-G, X-G>
%%       s.t.   X_ij  = e_ij     for (i,j) in (I_e, J_e)
%%              X_ij >= l_ij     for (i,j) in (I_l, J_l)
%%              X_ij <= u_ij     for (i,j) in (I_u, J_u)
%%              X    >= tau0*I   X is SDP (tau0>=0 and may be zero)
%%
%%%%%%%%%%%  Based on the algorithm  in  %%%%%%%%%%%%%%%%%%%
%%% "Calibrating Least Squares Semidefinite Programming  
%%%  with Equality and Inequality Constraints", National University of  Singapore
%%%   (June 2008 and revised in June 2009)
%%%%%%%%%  By Yan Gao and Defeng Sun  %%%%%%%%%%%%%%%%%%%%%%
  
%   Parameters:
%   Input
%   G         the given symmetric matrix
%   ConstrA:
%        e       the right hand side of equality constraints
%        I_e     row indices of the fixed elements
%        J_e     column indices of the fixed elements
%        l       the right hand side of lower bound constraint
%        I_l     row indices of the lower bound elements
%        J_l     column indices of the lower bound elements
%        u       the right hand side of upper bound constraint
%        I_u     row indices of the upper bound elements
%        J_u     column indices of the upper bound elements
%   OPTIONS   parameters in the OPTIONS structure
%   z0        the initial guess of dual variables
%
%   Output
%   X         the optimal primal solution
%   z:  
%      z.e    the optimal dual solution to equality constraints
%      z.l    the optimal dual solution to lower bound constraints
%      z.u    the optimal dual solution to upper bound constraints
%   infos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Last modified on March 28, 2010. 
%%% Preprocessed by Accelerated Proximal Gradient/Projected Gradient is added.



%%
%%-----------------------------------------
%%% get constraints infos from constrA
%%-----------------------------------------
%%
e   = ConstrA.e; I_e = ConstrA.Ie; J_e = ConstrA.Je;
l   = ConstrA.l; I_l = ConstrA.Il; J_l = ConstrA.Jl;
u   = ConstrA.u; I_u = ConstrA.Iu; J_u = ConstrA.Ju;
k_e = length(e); k_l = length(l);  k_u = length(u);
k   = k_e + k_l + k_u;  n = length(G);

%%                                      
%%-----------------------------------------
%% get parameters from the OPTIONS structure. 
%%-----------------------------------------
%%
tau0           = 0;        % lower bound of minimum eigenvalue
tol            = 1.0e-6;   % termination tolerance
tolBiCG        = 1.0e-2;   % termination for BiCGStab
maxit          = 500; 
maxitsub       = 20;       % maximum iterations in line search
maxitBiCG      = 200;      % maximum iterations in BiCGStab
delta          = 0.5;      % decrease rate in step length
use_precond    = 1;        % use preconditioner
disp           = 1;        % =1 display 
name_smoothfun = 'Huber';  % Huber smoothfun  %%%name_smoothfun = 'Smale'; 
use_ProjGrad   = 0;
use_APG        = 0;
eps_bar        = 1.0e1;    % eps_bar is in (0,+infty) 
tau            = 1;        % tau is in (0,1] and controls the decrease of epsilon
eta            = 0.5;
sigma          = 0.5e-6;          % decrease in the norm of merit function  
kappa          = 5.0e-3*k/n^2;    % modified on Aug 27, 2009. Regularized term
const_hist     = 5;
progress_test  = 1.0e-15;
if exist('OPTIONS')  
    if isfield(OPTIONS,'tau0');           tau0            = OPTIONS.tau0; end
    if isfield(OPTIONS,'tol');            tol             = OPTIONS.tol; end
    if isfield(OPTIONS,'tolBiCG');        tolBiCG         = OPTIONS.tolBiCG; end
    if isfield(OPTIONS,'maxit');          maxit           = OPTIONS.maxit; end
    if isfield(OPTIONS,'maxitsub');       maxitsub        = OPTIONS.maxitsub; end      
    if isfield(OPTIONS,'maxitBiCG');      maxitBiCG       = OPTIONS.maxitBiCG; end  
    if isfield(OPTIONS,'delta');          delta           = OPTIONS.delta; end    
    if isfield(OPTIONS,'use_precond');    use_precond     = OPTIONS.use_precond; end 
    if isfield(OPTIONS,'use_ProjGrad');   use_ProjGrad    = OPTIONS.use_ProjGrad; end 
    if isfield(OPTIONS,'use_APG');        use_APG         = OPTIONS.use_APG; end 
    if isfield(OPTIONS,'disp');           disp            = OPTIONS.disp; end        
    if isfield(OPTIONS,'name_smoothfun');  name_smoothfun  = OPTIONS.name_smoothfun; end        
    if isfield(OPTIONS,'eps_bar');         eps_bar         = OPTIONS.eps_bar; end
    if isfield(OPTIONS,'tau');             tau             = OPTIONS.tau; end
    if isfield(OPTIONS,'eta');             eta             = OPTIONS.eta; end
    if isfield(OPTIONS,'sigma');           sigma           = OPTIONS.sigma; end
    if isfield(OPTIONS,'kappa');           kappa           = OPTIONS.kappa; end       
end
if ( use_ProjGrad || use_APG )    %%% for a large number of constraints
   eps_bar = min(1,eps_bar);
end
r        = 0.25/max(1,eps_bar);   %%% r in (0,1) and eta & r satisfy: sqrt(2)*max(eta,r*eps_bar)<1
r_const0 = r;


t0 = clock;

%%% reset the pars
for i = 1:k_e     
    G(I_e(i),J_e(i)) = e(i);
    if I_e(i) ~= J_e(i)
        G(J_e(i),I_e(i)) = e(i);
    end
end
G      = G - tau0*speye(n);   % reset G
G      = (G + G')/2;          % make G symmetric
Ind    = find(I_e == J_e);    % reset diagonal part of e 
e(Ind) = e(Ind) - tau0;


k1        = 0;
f_eval    = 0;
num_BiCG  = 0;
prec_time = 0;
eig_time  = 0;
BiCG_time = 0;
phi_time  = 0;
rhs_time  = 0;
c        = 2.0*ones(k,1);     
prec     = ones(k,1);  
phi_hist = zeros(const_hist,1);
epsilon  = eps_bar;   % initial epsilon
epsilon0 = epsilon;

%%% initial value
if use_APG
    opts.maxit = min(ceil(k/n),20);
    %opts.maxit = 200;
    W = speye(n);
    if ( nargin == 4 )
        [X,z] = Pre_APGMex(G,W,ConstrA,opts,z0);
    else
        [X,z] = Pre_APGMex(G,W,ConstrA,opts);
    end
    z_e = z.e;
    z_l = z.l;
    z_u = z.u; 
elseif use_ProjGrad
    opts.maxit = min(ceil(k/n),20);
    %opts.maxit = 200;
    W = speye(n);
    if ( nargin == 4 )
        [X,z] = Pre_ProjGradMex(G,W,ConstrA,opts,z0);
    else
        [X,z] = Pre_ProjGradMex(G,W,ConstrA,opts);
    end
    z_e = z.e;
    z_l = z.l;
    z_u = z.u;
else
    if ( nargin == 4 )
        z_e = z0.e;
        z_l = z0.l;
        z_u = z0.u;
    else
        z_e = zeros(k_e,1);
        z_l = zeros(k_l,1);
        z_u = zeros(k_u,1);
    end
end 
x0_ze = z_e;
x0_zl = z_l;
x0_zu = z_u;


if disp
    fprintf('\n ******************************************************** \n')
    fprintf( '    The Smoothing Newton Method with the BICGSTAB Solver       ')
    fprintf('\n ******************************************************** \n')
    if strcmp(name_smoothfun,'Huber')
        fprintf('     ^^^Huber smoothing function^^^      ')
    elseif strcmp(name_smoothfun,'Smale')
        fprintf('     ^^^Smale smoothing function^^^      ')
    end
    fprintf('\n The information of this problem is as follows: \n')
    fprintf(' Dim. of    sdp      constr  = %d \n',n)
    fprintf(' Num. of equality    constr  = %d \n',k_e)
    fprintf(' Num. of lower bound constr  = %d \n',k_l)
    fprintf(' Num. of upper bound constr  = %d \n',k_u)
    fprintf(' The lower bounds: [ %2.1f, %2.1f ] \n',min(l),max(l))
    fprintf(' The upper bounds: [ %2.1f, %2.1f ] \n',min(u),max(u))
end


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
eig_time   = eig_time + etime(clock,t1);
 
t1 = clock;
[phi,g_ze,g_zl,g_zu,fe,fx] = Phi(name_smoothfun,epsilon,e,z_e,I_e,J_e,...
    l,z_l,I_l,J_l,u,z_u,I_u,J_u,P,lambda,kappa,c);  % the initial norm of the merit function
phi_time = phi_time + etime(clock,t1);
f_eval = f_eval + 1;
phi_hist(1) = phi^0.5;

if disp
    tt = etime(clock,t0);
    [hh,mm,ss] = time(tt);
    fprintf('\n   Iter   NumBiCGs    StepLen    SmoothPar    NormSmoothFun   time_used ')
    fprintf('\n    %2.0f       %s          %s        %3.2e       %3.2e       %d:%d:%d ',0,'-','-',epsilon,phi^0.5,hh,mm,ss)
end

while ( phi^0.5>tol && k1<maxit )
        
    theta     = r*min(1,phi^((1+tau)/2));                                         
    delta_eps = -epsilon + eps_bar*theta;    % compute delta_eps first
    
    t1 = clock;    
    [b_ze,b_zl,b_zu] = rhs(name_smoothfun,epsilon,delta_eps,e,I_e,J_e,z_e,g_ze,...
    l,I_l,J_l,z_l,g_zl,u,I_u,J_u,z_u,g_zu,fe,fx,P,lambda,kappa,c);  % to compute the right hand side term
    rhs_time = rhs_time + etime(clock,t1); 
    
    b_z = [b_ze;b_zl;b_zu];
          
    Omega = omega_mat(name_smoothfun,epsilon,lambda);
    
    % compute preconditioner
    if (use_precond)
    t2   = clock;  
    prec = precond_matrix(name_smoothfun,epsilon,e,I_e,J_e,l,I_l,J_l,u,I_u,J_u,fx,Omega,P,kappa,c);
    prec_time = prec_time + etime(clock,t2);
    end  
   
    % CG/BICGStab starts
    t3 = clock;
    if k_l+k_u == 0       % if no ineqalities, use CGs; otherwise bicgstab
        [d_z, flag, relres, iterk] = pre_cg(b_z,name_smoothfun,epsilon,e,I_e,J_e,...
            l,I_l,J_l,u,I_u,J_u,fx,Omega,P,maxitBiCG,tolBiCG,kappa,phi,eta,c,prec);
        if ( k1==0 && disp )
            fprintf( '\n $$$ No inequality constraints. BiCGStab is replaced by CG $$$ ')
        end
    else
        [d_z, relres, iterk, flag] = bicgstab(b_z,name_smoothfun,epsilon,e,I_e,J_e,...
            l,I_l,J_l,u,I_u,J_u,fx,Omega,P,maxitBiCG,tolBiCG,kappa,phi,eta,c,prec);
    end
    BiCG_time = BiCG_time + etime(clock,t3);
    num_BiCG  = num_BiCG + iterk;
    
    d_ze = d_z(1:k_e);
    d_zl = d_z(k_e+1:k_e+k_l);
    d_zu = d_z(k_e+k_l+1:k);
        
    z_e = x0_ze + d_ze;                           % temporary z_e
    z_l = x0_zl + d_zl;                           % temporary z_l
    z_u = x0_zu + d_zu;                           % temporary z_u
    epsilon = epsilon0 + delta_eps;               % temporary epsilon
    
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
    eig_time   = eig_time + etime(clock,t1);
         
    phi0 = phi;
    t1 = clock;
    [phi,g_ze,g_zl,g_zu,fe,fx] = ...
        Phi(name_smoothfun,epsilon,e,z_e,I_e,J_e,l,z_l,I_l,J_l,u,z_u,I_u,J_u,P,lambda,kappa,c); 
    phi_time = phi_time + etime(clock,t1); 
    f_eval   = f_eval + 1;
    
    % Line Search 
    k_inner=0;
    while( k_inner<=maxitsub && phi>(1-2*sigma*(1-sqrt(2)*max(eta,r*eps_bar))*delta^k_inner)*phi0 + 1.0e-12 )       
        k_inner = k_inner+1;
        z_e = x0_ze + delta^k_inner*d_ze;
        z_l = x0_zl + delta^k_inner*d_zl;
        z_u = x0_zu + delta^k_inner*d_zu;
        epsilon = epsilon0 + delta^k_inner*delta_eps;
        
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
        eig_time   = eig_time + etime(clock,t1);
                
        t1 = clock;
        [phi,g_ze,g_zl,g_zu,fe,fx] = ...
          Phi(name_smoothfun,epsilon,e,z_e,I_e,J_e,l,z_l,I_l,J_l,u,z_u,I_u,J_u,P,lambda,kappa,c); 
        phi_time = phi_time + etime(clock,t1); 
    end  
    
    
    %%% update r 
    if  phi^0.5 < 5.0e0
        r = r_const0/10;
    end
    %%%
            
    k1     = k1 + 1;
    f_eval = f_eval + k_inner;
    
    x0_ze = z_e;
    x0_zl = z_l;
    x0_zu = z_u;
    epsilon0 = epsilon;
    
    if disp
        tt = etime(clock,t0);
        [hh,mm,ss] = time(tt);
        fprintf('\n   %2.0d       %2.0d         %3.2f      %3.2e       %3.2e       %d:%d:%d ',...
            k1,iterk,delta^k_inner,epsilon,phi^0.5,hh,mm,ss)
    end

    % slow convergence test
    if  k1<const_hist
        phi_hist(mod(k1,const_hist)+1) = phi^0.5;
    else
        for i=1:const_hist-1
            phi_hist(i) = phi_hist(i+1);
        end
        phi_hist(const_hist) = phi^0.5;
    end  
    if ( k1 >= const_hist-1 && phi_hist(1)-phi_hist(const_hist) <= progress_test )
        fprintf('\n Warning: Progress is too slow! :( ')
        break
    end
        
end  % End of outer loop

% correct the final z_l & z_u
g_zl = g_zl - kappa*epsilon*z_l;
z_l  = z_l - g_zl;
g_zu = g_zu - kappa*epsilon*z_u;
z_u  = z_u - g_zu;

% optimal solution X*
s  = smoothing_fun(name_smoothfun,epsilon,lambda);   
Ip = find(s>1.0e-8);
r1 = length(Ip);

if (r1==0)
    X = zeros(n,n);
else
    if (r1<n/2)
        s1 = s(Ip);
        s1 = s1.^0.5;
        P1 = P(:,1:r1);
        if r1>1
            P1 = P1*sparse(diag(s1));
            X  = P1*P1';  % Optimal solution X*
        else
            X = s1^2*P1*P1';
        end
    else
        In = find(s~=lambda);
        r2 = length(In);
        if (r2>0)               %%if r2 =0, then X*=X;
            s2 = s(In)-lambda(In);
            s2 = s2.^0.5;
            P2 = P(:,n-r2+1:n);
            if r2 >1
                P2 = P2*sparse(diag(s2));
                X = X + P2*P2';  % Optimal solution X*
            else
                X = X + s2^2*P2*P2';
            end
        end
    end
end
X = (X + X')/2;

% min_abs_lambda = min(abs(lambda));
r_X = r1;        % rank of X*
r_Z = length(find(abs(s-lambda)>1.0e-8));  % rank of Z
%lambda_min = min(abs(lambda));

% optimal primal and negative dual value
prim_val = sum(sum((X-G).*(X-G)))/2;
dual_val = sum(sum(X.*X)) - sum(sum(G.*G));
dual_val = dual_val/2 - z_e'*e - z_l'*l + z_u'*u;


% convert to original X*
X = X + tau0*eye(n);
z.e = z_e;
z.l = z_l;
z.u = z_u;
%%%%%%%%%%%%%%%%%%%%%%%

info.P         = P;
info.lam       = s;
info.rank      = r_X;
info.numIter   = k1;
info.numBiCG   = num_BiCG;
info.numEig    = f_eval;
info.eigtime   = eig_time;
info.BiCGtime  = BiCG_time;
info.prectime  = prec_time;
info.dualVal   = -dual_val;
time_used = etime(clock,t0);
if disp==1    
    %fid = fopen('result.txt','wt');
    %fprintf(fid,'\n');
    fprintf('\n\n ================ Final Information ================= \n');
    fprintf(' Total number of iterations      = %2.0f \n', k1);
    fprintf(' Number of func. evaluations     = %2.0f \n', f_eval);
    fprintf(' Number of BiCGStabs             = %2.0f \n', num_BiCG);
    fprintf(' Primal objective value          = %d \n', prim_val);
    fprintf(' Dual objective value            = %d \n', -dual_val);
    fprintf(' Norm of smoothing func          = %3.2e \n', phi^0.5);
    fprintf(' Rank of  X*-(tau0*I)            === %2.0d \n', r_X);
    fprintf(' Rank of optimal multiplier Z*   === %2.0d \n', r_Z);
    fprintf(' Computing time for precond            = %3.1f \n', prec_time);
    fprintf(' Computing time for BiCGStab           = %3.1f \n', BiCG_time);
    fprintf(' Computing time for eigen-decom        = %3.1f \n', eig_time);
    fprintf(' Computing time for the merit fun.     = %3.1f \n', phi_time);
    fprintf(' Computing time for the rhs            = %3.1f \n', rhs_time);
    fprintf(' Total computing time (secs)           = %3.1f \n', time_used);
    fprintf(' ====================================================== \n');
    %fclose(fid);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% end of the main program %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





 


%%  **************************************
%%  ******** All Sub-routines  ***********
%%  **************************************

%%% To change the format of time 
function [h,m,s] = time(t)
t = round(t); 
h = floor(t/3600);
m = floor(rem(t,3600)/60);
s = rem(rem(t,60),60);
%%% End of time.m

 



%%% mexeig decomposition
function [P,lambda] = MYmexeig(X)
[P,lambda] = mexeig(X);
%[P,lambda] = eig(X);
%lambda=diag(lambda);
P          = real(P);
lambda     = real(lambda);
if issorted(lambda)
    lambda = lambda(end:-1:1);
    P      = P(:,end:-1:1);
elseif issorted(lambda(end:-1:1))
    return;
else
    [lambda, Inx] = sort(lambda,'descend');
    P = P(:,Inx);
end
% % % Rearrange lambda and P in the nonincreasing order
% % if lambda(1) < lambda(end) 
% %     lambda = lambda(end:-1:1);
% %     P      = P(:,end:-1:1);
% % end
return
%%% End of MYmexeig.m




%%% To generate the uniform smoothing function 
function s = smoothing_fun(name_smoothfun,epsilon,lambda)
n = length(lambda);
s = zeros(n,1);

switch name_smoothfun
    case 'Huber'
        for i=1:n
            if lambda(i) >= 0.5*epsilon
                s(i) = lambda(i);
            elseif lambda(i) > -0.5*epsilon
                s(i) = 1/(2*epsilon)*(lambda(i)+0.5*epsilon)^2;
            end
        end
    case 'Smale'
        for i=1:n
            if lambda(i)>0
                s(i) = 0.5*(sqrt(lambda(i)^2+epsilon^2)+lambda(i));
            else
                s(i) = 0.5*epsilon^2/(sqrt(lambda(i)^2+epsilon^2)-lambda(i));
            end
        end
end
return
%%% End of smoothing.m 
 




%%% To generate partial derivatives of G(epsilon,y) wrt epsilon and y 
function  [fe fx] = smoothing_ff(name_smoothfun,epsilon,x)
n  = length(x);
fe = zeros(n,1);
fx = zeros(n,1);

switch name_smoothfun
    case 'Huber'
        for i=1:n
            if x(i)>=0.5*epsilon
                fe(i) = 0;
                fx(i) = 1;
            elseif x(i)>-0.5*epsilon
                fe(i) = 1/8 - 0.5*(x(i)/epsilon)^2;
                fx(i) = 0.5 + x(i)/epsilon;
            end
        end
    case 'Smale'
        for i=1:n
            if x(i)>0
                fx(i) = 0.5*(1+x(i)/sqrt(epsilon^2+x(i)^2));
            else
                fx(i) = 0.5*epsilon/sqrt(epsilon^2+x(i)^2);
                fx(i) = fx(i)*epsilon/(sqrt(epsilon^2+x(i)^2)-x(i));
            end
            fe(i) = 0.5*epsilon/sqrt(epsilon^2+x(i)^2);
        end
end
return
%%% End of smoother_ff.m





%%% To generate the first order difference matrix 
function Omega = omega_mat(name_smoothfun,epsilon,lambda)
n = length(lambda);

switch name_smoothfun
    case 'Huber'
        idx.b = find(lambda>=0.5*epsilon);
        idx.s = find(lambda<=-0.5*epsilon);
        idx.m = setdiff([1:n],[idx.s;idx.b]);

        r.b = length(idx.b); 
        r.m = length(idx.m); 
        r.s = length(idx.s); 
        Omega.dim = [r.b; r.m; r.s]; 

        if (r.b<n&&r.s<n)  %if r.b==n, Omega.bb=ones(n,n); if r.s==n, Omega.ss=zeros(n,n).
            db = lambda(1:r.b);
            dm = lambda(r.b+1:r.b+r.m);
            ds = lambda(r.b+r.m+1:n);
            tmp = 1/(2*epsilon)*(dm+0.5*epsilon).^2;

            Omega.mm = dm*ones(1,r.m)+ones(r.m,1)*dm';
            Omega.mm = 0.5*(Omega.mm./epsilon + ones(r.m,r.m));

            Omega.bm = db*ones(1,r.m)-ones(r.b,1)*tmp';
            Omega.bm = Omega.bm./( db*ones(1,r.m)-ones(r.b,1)*dm' );

            Omega.bs = db*ones(1,r.s);
            Omega.bs = Omega.bs./( db*ones(1,r.s)-ones(r.b,1)*ds' );

            Omega.ms = tmp*ones(1,r.s);
            Omega.ms = Omega.ms./( dm*ones(1,r.s)-ones(r.m,1)*ds' );           
        end
    case 'Smale'
    lambda_ep = sqrt(lambda.^2+epsilon^2);

    Omega = lambda*ones(1,n)+ones(n,1)*lambda';
    Omega = Omega./(lambda_ep*ones(1,n) + ones(n,1)*lambda_ep');
    Omega = 0.5*(ones(n,n) + Omega);
end
return
%%%% End of omega_mat.m 






%%% To generate the merit function phi 
function  [phi,g_ze,g_zl,g_zu,fe,fx] = ...
   Phi(name_smoothfun,epsilon,e,z_e,I_e,J_e,l,z_l,I_l,J_l,u,z_u,I_u,J_u,P,lambda,kappa,c)

n   = length(P);
k_e = length(e);
k_l = length(l);
k_u = length(u);
k   = k_e + k_l + k_u;

g_ze0 = zeros(k_e,1);
g_zl0 = zeros(k_l,1);
g_zu0 = zeros(k_u,1);

s = smoothing_fun(name_smoothfun,epsilon,lambda);

%%% to generate the smoothing function for the SDP projection operator
const_sparse = 2;  %sparsity parameter
if k <= const_sparse*n;   %sparse form
    M = P';
    i=1;
    while (i<=n)
        M(i,:) = s(i)*M(i,:);
        i=i+1;
    end

    i=1;
    while (i<=k_e)
        g_ze0(i) = P(I_e(i),:)*M(:,J_e(i));
        i = i+1;
    end
    i=1;
    while (i<=k_l)
        g_zl0(i) = P(I_l(i),:)*M(:,J_l(i));
        i=i+1;
    end
    i=1;
    while (i<=k_u)
        g_zu0(i) = -P(I_u(i),:)*M(:,J_u(i));
        i=i+1;
    end
    
else  %dense form    
    Ip = find(s>0);
    r  = length(Ip);
    
    if (r==0)
        M = zeros(n,n);
    else %%% cannnot use the complementary form!
        s1 = s(Ip);
        s1 = s1.^0.5;
        P1 = P(:,Ip);        
        if r>1
            P1 = P1*sparse(diag(s1));
            M  = P1*P1'; %
        else
            M = s1^2*P1*P1';
        end
    end

    i=1;
    while (i<=k_e)
        g_ze0(i) = M(I_e(i),J_e(i));
        i=i+1;
    end
    i=1;
    while (i<=k_l)
        g_zl0(i) = M(I_l(i),J_l(i));
        i=i+1;
    end
    i=1;
    while (i<=k_u)
        g_zu0(i) = -M(I_u(i),J_u(i));
        i=i+1;
    end
end

g_ze = c(1:k_e).*(g_ze0-e) + kappa*epsilon*z_e;

g_zl = z_l - c(k_e+1:k_e+k_l).*(g_zl0-l);
[fe.l,fx.l] = smoothing_ff(name_smoothfun,epsilon,g_zl);
g_zl = z_l - smoothing_fun(name_smoothfun,epsilon,g_zl) + kappa*epsilon*z_l;

g_zu = z_u - c(k_e+k_l+1:k).*(u + g_zu0);
[fe.u,fx.u] = smoothing_ff(name_smoothfun,epsilon,g_zu);
g_zu = z_u - smoothing_fun(name_smoothfun,epsilon,g_zu) + kappa*epsilon*z_u;

phi = epsilon^2 + norm(g_ze)^2 + norm(g_zl)^2 + norm(g_zu)^2;
return
%%% End of Phi.m






%%% To generate the right hand side of the linear system
function  [b_ze,b_zl,b_zu] = rhs(name_smoothfun,epsilon,delta_eps,e,I_e,J_e,z_e,g_ze,...
    l,I_l,J_l,z_l,g_zl,u,I_u,J_u,z_u,g_zu,fe,fx,P,lambda,kappa,c)
  
n   = length(P);
k_e = length(e);
k_l = length(l);
k_u = length(u);
k   = k_e +k_l + k_u;

b_ze = -g_ze;
b_zl = -g_zl;
b_zu = -g_zu;

s  = smoothing_ff(name_smoothfun,epsilon,lambda);
Ip = find(abs(s)>0);
r  = length(Ip);

const_sparse = 2;  
if (r>0)
    s1 = s(Ip);
    P1 = P(:,Ip);    
    if ( k <= const_sparse*n )
        D = P1';
        i=1;
        while (i<=r)
            D(i,:) = s1(i)*D(i,:);
            i=i+1;
        end

        %%% to generate the first term b_ze
        i =1;
        while (i<=k_e)
            b_ze(i) = b_ze(i) - c(i)*delta_eps*( P1(I_e(i),:)*D(:,J_e(i)) );
            i=i+1;
        end
        %%% to generate the second term b_zl
        i=1;
        while (i<=k_l)
            b_zl(i) = b_zl(i) - c(k_e+i)*delta_eps*fx.l(i)*( P1(I_l(i),:)*D(:,J_l(i)) );
            i=i+1;
        end
        %%% to generate the second term b_zu
        i=1;
        while (i<=k_u)
            b_zu(i)= b_zu(i) - c(k_e+k_l+i)*delta_eps*fx.u(i)*( -P1(I_u(i),:)*D(:,J_u(i)) );
            i=i+1;
        end
        
    else %dense case
        %D = P*diag(s)*P';
        
        s1 = s1.^0.5;        
        if r>1
            P1 = P1*sparse(diag(s1));
            D  = P1*P1';
        else
            D= s1^2*P1*P1';
        end
 
        %%% to generate the first term b_ze
        i=1;
        while(i<=k_e)
            b_ze(i) = b_ze(i) - c(i)*delta_eps*D(I_e(i),J_e(i));
            i=i+1;
        end
        %%% to generate the second term b_zl
        i=1;
        while (i<=k_l)
            b_zl(i) = b_zl(i) - c(k_e+i)*delta_eps*fx.l(i)*D(I_l(i),J_l(i));
            i=i+1;
        end
        %%% to generate the second term b_zu
        i=1;
        while (i<=k_u)
            b_zu(i) = b_zu(i) - c(k_e+k_l+i)*delta_eps*fx.u(i)*( -D(I_u(i),J_u(i)) );
            i=i+1;
        end
    end
end
b_ze = b_ze - kappa*delta_eps*z_e;
b_zl = b_zl + delta_eps*fe.l - kappa*delta_eps*z_l;
b_zu = b_zu + delta_eps*fe.u - kappa*delta_eps*z_u;
return
%%% End of rhs.m



%%% To generate the Jacobain product F'(y)(x) 
function [Ax_ze, Ax_zl, Ax_zu] = Jacobian_matrix(name_smoothfun,epsilon,x_ze,x_zl,x_zu,...
    e,I_e,J_e,l,I_l,J_l,u,I_u,J_u,fx,Omega,P,kappa,c)
 
n   = length(P);
k_e = length(e);
k_l = length(l);
k_u = length(u);
k   = k_e + k_l + k_u;

Ax_ze = zeros(k_e,1);
Ax_zl = zeros(k_l,1);
Ax_zu = zeros(k_u,1);
 
%%%% Dense form is preferred.
const_sparse = 2; 
switch name_smoothfun
    case 'Huber'
        r.b = Omega.dim(1);
        r.m = Omega.dim(2);
        r.s = Omega.dim(3);
        if r.s==n
            Ax_ze = (kappa*epsilon + 1.0e-10)*x_ze;
            Ax_zl = (ones(k_l,1) - fx.l).*x_zl + (kappa*epsilon + 1.0e-10)*x_zl;
            Ax_zu = (ones(k_u,1) - fx.u).*x_zu + (kappa*epsilon + 1.0e-10)*x_zu;
        elseif r.b==n
            Ax_ze = c(1:k_e).*(0.5*x_ze);
            Ax_zl = c(k_e+1:k_e+k_l).*fx.l.*(0.5*x_zl);
            Ax_zu = c(k_e+k_l+1:k).*fx.u.*( 0.5*x_zu );

            Ind = find(I_e==J_e);                        
            Ax_ze(Ind) = 2*Ax_ze(Ind);
          
            Ind = find(I_l==J_l);                        
            Ax_zl(Ind) = 2*Ax_zl(Ind);           
          
            Ind = find(I_u==J_u);                        
            Ax_zu(Ind) = 2*Ax_zu(Ind);
            
            Ax_ze = Ax_ze + (kappa*epsilon+1.0e-10)*x_ze;
            Ax_zl = Ax_zl + (ones(k_l,1)-fx.l).*x_zl + (kappa*epsilon + 1.0e-10)*x_zl;
            Ax_zu = Ax_zu + (ones(k_u,1)-fx.u).*x_zu + (kappa*epsilon + 1.0e-10)*x_zu;
            
                                
        else
            
            Z = zeros(n,n);
            for i=1:k_e
                Z(I_e(i), J_e(i)) = x_ze(i);
            end
            for i=1:k_l
                Z(I_l(i), J_l(i)) = x_zl(i) + Z(I_l(i), J_l(i));
            end
            for i=1:k_u
                Z(I_u(i), J_u(i)) = -x_zu(i) + Z(I_u(i), J_u(i));  %%% upper bound
            end
            Z = 0.5*(Z + Z');
            
            if (k<=const_sparse*n)   %sparse form
                %disp('sparse form')
                %H = [Omega.mat.*(P'*sparse(Z)*P)]*P'; 
                                                               
                if r.b <= r.s   %more zeros than ones in Omega.mat
                    P1 = P(:,1:r.b+r.m);
                    P2 = P(:,r.b+r.m+1:n);

                    U = P1'*sparse(Z);
                    Omega11 = [ones(r.b,r.b) Omega.bm; Omega.bm' Omega.mm];
                    Omega12 = [Omega.bs; Omega.ms];

                    H12 =  Omega12.*(U*P2);
                    H   = [(Omega11.*(U*P1))*P1' + H12*P2'; H12'*P1'];

                    %%% to generate the first part: Ax_ze
                    i=1;
                    while (i<=k_e)
                        Ax_ze(i) = c(i)*P(I_e(i),:)*H(:,J_e(i));
                        i=i+1;
                    end
                    %%% to generate the second part: Ax_zl
                    i=1;
                    while (i<=k_l)
                        Ax_zl(i) = c(k_e+i)*fx.l(i)*(P(I_l(i),:)*H(:,J_l(i)));
                        i= i+1;
                    end
                    %%% to generate the second part: Ax_zu
                    i=1;
                    while (i<=k_u)
                        Ax_zu(i) = c(k_e+k_l+i)*fx.u(i)*(-P(I_u(i),:)*H(:,J_u(i)));
                        i= i+1;
                    end
                else   %more ones than zeros in Omega.mat
                    %H = ( Omega_bar.mat.*(P'*sparse(Z)*P) )*P';
                    P1 = P(:,1:r.b);
                    P2 = P(:, r.b+1:n);

                    U = P2'*sparse(Z);
                    Omega_bar12 = [ones(r.b,r.m)-Omega.bm  ones(r.b,r.s)-Omega.bs];
                    Omega_bar22 = ones(r.m + r.s, r.m + r.s)- [Omega.mm Omega.ms; Omega.ms' zeros(r.s, r.s)];

                    H21 = (Omega_bar12'.*(U*P1));
                    H = [H21'*P2';  H21*P1'+(Omega_bar22.*(U*P2))*P2'];

                    i=1;
                    while (i<=k_e)
                        if (I_e(i)==J_e(i))
                            Ax_ze(i) = x_ze(i) - P(I_e(i),:)*H(:,J_e(i));
                        else
                            Ax_ze(i) = x_ze(i)*0.5 - P(I_e(i),:)*H(:,J_e(i));
                        end
                        Ax_ze(i) = c(i)*Ax_ze(i);
                        i=i+1;
                    end
                    %%% to generate the second part: Ax_zl
                    i=1;
                    while (i<=k_l)
                        if (I_l(i)==J_l(i))
                            Ax_zl(i) = x_zl(i) - P(I_l(i),:)*H(:,J_l(i));
                        else
                            Ax_zl(i) = x_zl(i)*0.5 - P(I_l(i),:)*H(:,J_l(i));
                        end
                        Ax_zl(i) = c(k_e+i)*fx.l(i)*Ax_zl(i);
                        i= i+1;
                    end
                    %%% to generate the second part: Ax_zu
                    i=1;
                    while (i<=k_u)  %there is a negative sign for the upper bound case
                        if(I_u(i)==J_u(i))
                            Ax_zu(i) = - (-x_zu(i) - P(I_u(i),:)*H(:,J_u(i)));
                        else
                            Ax_zu(i) = x_zu(i)*0.5 - (-P(I_u(i),:)*H(:,J_u(i)));
                        end                        
                        Ax_zu(i) = c(k_e+k_l+i)*fx.u(i)*Ax_zu(i);
                        i= i+1;
                    end
                end

            else %dense form
                %disp('dense form')
                %H = P*[Omega.mat.*(P'*(Z)*P)]*P';
                %Z =  full(Z);
                              
                if r.b <= r.s   %more zeros than ones in Omega.mat
                    P1 = P(:,1:r.b+r.m);
                    P2 = P(:,r.b+r.m+1:n);

                    U = P1'*Z;
                    Omega11 = [ones(r.b,r.b) Omega.bm; Omega.bm' Omega.mm];
                    Omega12 = [Omega.bs; Omega.ms];

                    H = P1*( (Omega11.*(U*P1))*P1' + 2.0*(Omega12.*(U*P2))*P2');
                    H = (H + H')/2;
                else   %% more ones than zeros in Omega.mat
                    %H =  Z  - P*( Omega_bar.mat.*(P'*sparse(Z_e+Z_l-Z_u)*P) )*P';                    
                    P1 = P(:,1:r.b);
                    P2 = P(:, r.b+1:n);

                    U = P2'*Z;
                    Omega_bar12 = [ones(r.b,r.m)-Omega.bm  ones(r.b,r.s)-Omega.bs];
                    Omega_bar22 = ones(r.m+r.s, r.m+r.s) - [Omega.mm Omega.ms; Omega.ms' zeros(r.s, r.s)];

                    H = P2*( (Omega_bar22.*(U*P2))*P2' + 2.0*(Omega_bar12'.*(U*P1))*P1' );
                    H = (H + H')/2;
                    H = Z - H;
                end
                %%% to generate the first part Ax_ze               
                i=1;
                while (i<=k_e)
                    Ax_ze(i) = c(i)*H(I_e(i),J_e(i));
                    i=i+1;
                end
                %%% to generate the second part: Ax_zl
                i=1;
                while (i<=k_l)
                    Ax_zl(i) = c(k_e+i)*fx.l(i)*H(I_l(i),J_l(i));
                    i=i+1;
                end
                %%% to generate the second part: Ax_zu
                i=1;
                while (i<=k_u)
                    Ax_zu(i) = c(k_e+k_l+i)*fx.u(i)*(-H(I_u(i),J_u(i)));
                    i=i+1;
                end            
            end
            Ax_ze = Ax_ze + (kappa*epsilon+1.0e-10)*x_ze;
            Ax_zl = Ax_zl + (ones(k_l,1)-fx.l).*x_zl + (kappa*epsilon+1.0e-10)*x_zl;
            Ax_zu = Ax_zu + (ones(k_u,1)-fx.u).*x_zu + (kappa*epsilon+1.0e-10)*x_zu;         
        end  

    case 'Smale'
        Z = zeros(n,n);
        for i=1:k_e
            Z(I_e(i), J_e(i)) = x_ze(i);
        end
        for i=1:k_l
            Z(I_l(i), J_l(i)) = x_zl(i) + Z(I_l(i), J_l(i));
        end
        for i=1:k_u
            Z(I_u(i), J_u(i)) = -x_zu(i) + Z(I_u(i), J_u(i));  %%% upper bound
        end
        Z = 0.5*(Z + Z');
        
        if k <= const_sparse*n %sparse form
            % disp('sparse form')
            H = P'*sparse(Z)*P;         %H=P^T*[Z]*P
            H = Omega.*H;
            H = H*P';
            %%% to generate the first part: Ax_ze
            i=1;
            while (i<=k_e)
                Ax_ze(i) = c(i)*P(I_e(i),:)*H(:,J_e(i));
                i=i+1;
            end
            %%% to generate the second part: Ax_zl
            i=1;
            while (i<=k_l)
                Ax_zl(i) = c(k_e+i)*fx.l(i)*(P(I_l(i),:)*H(:,J_l(i)));
                i= i+1;
            end
            %%% to generate the second part: Ax_zu
            i=1;
            while (i<=k_u)
                Ax_zu(i) = c(k_e+k_l+i)*fx.u(i)*(-P(I_u(i),:)*H(:,J_u(i)));
                i= i+1;
            end
        else %dense form
            %disp('dense form')
            %Z = full(Z);
            H = P'*Z*P;
            H = Omega.*H;
            H = H*P';
            H = P*H;

            %%% to generate the first part Ax_ze
            i=1;
            while (i<=k_e)
                Ax_ze(i) = c(i)*H(I_e(i),J_e(i));
                i=i+1;
            end
            %%% to generate the second part: Ax_zl
            i=1;
            while (i<=k_l)
                Ax_zl(i) = c(k_e+i)*fx.l(i)*H(I_l(i),J_l(i));
                i=i+1;
            end
            %%% to generate the second part: Ax_zu
            i=1;
            while (i<=k_u)
                Ax_zu(i) = c(k_e+k_l+i)*fx.u(i)*(-H(I_u(i),J_u(i)));
                i=i+1;
            end
        end
        Ax_ze = Ax_ze + (kappa*epsilon+1.0e-10)*x_ze;
        Ax_zl = Ax_zl + (ones(k_l,1)-fx.l).*x_zl + (kappa*epsilon + 1.0e-10)*x_zl;
        Ax_zu = Ax_zu + (ones(k_u,1)-fx.u).*x_zu + (kappa*epsilon + 1.0e-10)*x_zu;
        
end  %end of switch  
return
%%% End of Jacobian_matrix.m 
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% PCG method %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This is exactly the algorithm given by Hestenes and Stiefel (1952)
%% An iterative method to solve A(x) =b  
%% The symmetric positive definite matrix M is a preconditioner for A.
%% See Pages 527 and 534 of Golub and va Loan (1996)
function [p,flag,relres,iterk] = pre_cg( b,name_smoothfun,epsilon,e,I_e,J_e,...
    l,I_l,J_l,u,I_u,J_u,fx,Omega,P,maxit,tol,kappa,phi,eta,c,prec )

k_e = length(e);
k_l = length(l);
k_u = length(u);
k   = k_e + k_l + k_u;
n = length(P);
p = zeros(k,1);

r = b;  
n2b = norm(b); 
if n2b <= 1.0e-12
    iterk  = 0;
    relres = 0;         
    flag   = 0;
    return;
end
if n2b > 1.0e2
    maxit = min(1,maxit);
end

tol  = max(tol, min(5.0e-2,n2b)); 
tol  = min(tol, eta*phi^0.5/n2b );
tolb = tol*n2b;     % relative tolerance 

flag   = 1;
iterk  = 0;
relres = 1000;  %give a big value on relres

%%% preconditioning 
z   = r./prec;      % z = M\r; here M = diag(c); if M is not the identity matrix 
rz1 = r'*z; 
rz2 = 1; 
d   = z;

%%% CG iteration
for k1 = 1:maxit
   if k1 > 1
       beta = rz1/rz2;
       d    = z + beta*d;
   end
  
   x_ze = d(1:k_e);
   x_zl = d(k_e+1:k_e+k_l);
   x_zu = d(k_e+k_l+1:k);
   
   w = Jacobian_matrix(name_smoothfun,epsilon,x_ze,x_zl,x_zu,...
       e,I_e,J_e,l,I_l,J_l,u,I_u,J_u,fx,Omega,P,kappa,c);
  
  if ( k > n )
        w = w + 1.0e-2*min(1,0.1*n2b)*d; % perturb it to avoid singularity
   end
   
   denom = d'*w;
   iterk = k1;
   relres = norm(r)/n2b;              % relative residue =norm(r) / norm(b)
 
   if denom <= 0 
       % sssss = 0;
       p = d/norm(d);                 % d is not a descent direction
       break;  % exit
   else
       alpha = rz1/denom;
       p = p + alpha*d;
       r = r - alpha*w;
   end
   z = r./prec;                      
   if norm(r) <= tolb                % Exit if Hp=b is solved within the relative tolerance
       iterk = k1;
       relres = norm(r)/n2b;         % relative residue = norm(r)/norm(b)
       flag = 0;
       break
   end
   rz2 = rz1;
   rz1 = r'*z;
end
return
%%% End of pre_cg.m





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% BICGStab method %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, relres, iter, flag] = bicgstab(b_z,name_smoothfun,epsilon,e,I_e,J_e,...
    l,I_l,J_l,u,I_u,J_u,fx,Omega,P,maxit,tol,kappa,phi,eta,c,prec)

%% BICGStab solves a linear system using the biconjugate gradient
%% stabilized method developed by 
%% H.A. van der Vorst,
%% "BI-CGSTAB: A fast and smoothly converging variant of BI-CG for the solution of nonsymmetric linear systems", 
%% SIAM Journal on Scientific and Statistical Computing 13 (1992), pp.631-644. 
%%
%    Output:
%    x               the optimal solution
%    relres          the relative norm of error
%    iter            the number of iterations needed
%    flag            the return flag
%                    ---  0 = the solution is found within the specified tolerance
%                    ---  1 = a satisfactory solution is not found and the iteration limit is exceeded
%                    --- -1 = the method broke down with RHO = 0 
%                    --- -2 = the method broke down with OMEGA = 0 
n   = length(P);
k_e = length(e);
k_l = length(l);
k_u = length(u);
k   = k_e+k_l+k_u;

% intial x0=0
x = zeros(k,1);

omeg = 1.0;
flag = 1;
bnrm2 = norm(b_z);

if ( bnrm2 <= 1.0e-12 )
    iter = 0;
    relres = 0;         
    flag = 0;
    return;
end

r      = b_z;
r_tld  = r;
relres = norm(r)/bnrm2;

tol  = max(tol, min(5.0e-2,bnrm2)); 
tolb = min(eta*phi^0.5, tol*bnrm2);
%tol = min(tol, eta*phi^0.5/bnrm2);

smtol = 1e-40;
for iter = 1:maxit
    rho = r_tld'*r;
    if ( abs(rho) < smtol )
        flag = -1;
        break;
    end
    if (1<iter)
        beta = (rho/rho_1)*(alpha0/omeg);
        pp = r + beta*(pp - omeg*v);                
    else 
        pp = r;           
    end
    p_hat = pp./prec;
      
    x_ze = p_hat(1:k_e);
    x_zl = p_hat(k_e+1:k_e+k_l);
    x_zu = p_hat(k_e+k_l+1:k);
 
    [Ax_ze, Ax_zl, Ax_zu] = Jacobian_matrix(name_smoothfun,epsilon,x_ze,x_zl,x_zu,...
        e,I_e,J_e,l,I_l,J_l,u,I_u,J_u,fx,Omega,P,kappa,c);   
    v = [Ax_ze; Ax_zl; Ax_zu];
    
    alpha0 = rho/(r_tld'*v);
    s      = r - alpha0*v;

    % early convergence check
    if (norm(s) < tolb)
        flag = 0;
        x = x + alpha0*p_hat;
        relres = norm(s)/bnrm2;        
        break;
    end

    s_hat = s./prec;
    x_ze = s_hat(1:k_e);
    x_zl = s_hat(k_e+1:k_e+k_l);
    x_zu = s_hat(k_e+k_l+1:k);
 
    [Ax_ze, Ax_zl, Ax_zu] = Jacobian_matrix(name_smoothfun,epsilon,x_ze,x_zl,x_zu,...
        e,I_e,J_e,l,I_l,J_l,u,I_u,J_u,fx,Omega,P,kappa,c);   
    t = [Ax_ze; Ax_zl; Ax_zu];
   
     
    omeg = (t'*s)/(t'*t);

    %%% update x
    x = x + alpha0*p_hat + omeg*s_hat;
    
    %%% check convergence
    r     = s - omeg*t;
    rho_1 = rho;
    relres = norm(r)/bnrm2;
    
    if ( norm(r) <= tolb )        
        flag = 0;
        break;
    end
    if ( abs(omeg) < smtol )
        flag = -2;
        break;
    end    
end
return
%%% End of bicgstab.m 
  





%%% To generate the (approximate) diagonal preconditioner 
function prec = precond_matrix(name_smoothfun,epsilon,e,I_e,J_e,l,I_l,J_l,u,I_u,J_u,fx,Omega,P,kappa,c)

n   = length(P);
k_e = length(e);
k_l = length(l);
k_u = length(u);
k   = k_e + k_l + k_u;

prec = ones(k,1);

const_prec = 1; 
switch name_smoothfun
    case 'Huber'
        r.b = Omega.dim(1);
        r.m = Omega.dim(2);
        r.s = Omega.dim(3);

        if (r.s == n)
            prec(1:k_e)         = kappa*epsilon*ones(k_e,1);
            prec(k_e+1:k_e+k_l) = (1-fx.l) + kappa*epsilon*ones(k_l,1);
            prec(k_e+k_l+1:k)   = (1-fx.u) + kappa*epsilon*ones(k_u,1);
            Ind       = find( prec<=1.0e-8 );
            prec(Ind) = 1.0e-8;
        else
            H = P';
            H = H.*H;            
            if (k <= const_prec*n && r.b<n)   % compute the exact diagonal preconditioner
                % the first part: prec_ze
                Ind = find(I_e~=J_e);
                k1  = length(Ind);
                if (k1>0)
                    H1 = zeros(n,k1);
                    for i=1:k1
                        H1(:,i) = P(I_e(Ind(i)),:)'.*P(J_e(Ind(i)),:)';
                    end
                end
                
                if (r.b<r.s)
                    Omega11 = [ones(r.b,r.b) Omega.bm; (Omega.bm)'  Omega.mm];
                    Omega11_1 = Omega11* H(1:r.b+r.m,:);
                                        
                    Omega12 = [Omega.bs;Omega.ms];                    
                    H12  = H(1:r.b+r.m,:)'*Omega12;
                    
                    if(k1>0)
                        H12_1 = H1(1:r.b+r.m,:)'*Omega12;
                        Omega11_2 = Omega11* H1(1:r.b+r.m,:);
                    end
                    
                    j = 0;
                    for i=1:k_e
                        if (I_e(i)==J_e(i))
                            prec(i) = H(1:r.b+r.m,I_e(i))'*Omega11_1(:,J_e(i));
                            prec(i) = prec(i) + 2.0*(H12(I_e(i),:)*H(r.b+r.m+1:n,J_e(i))) + kappa*epsilon;
                        else
                            j = j+1;
                            prec(i) = H(1:r.b+r.m,I_e(i))'*Omega11_1(:,J_e(i));
                            prec(i) = prec(i) + 2.0*(H12(I_e(i),:)*H(r.b+r.m+1:n,J_e(i)));
                            prec(i) = prec(i) + H1(1:r.b+r.m,j)'*Omega11_2(:,j);
                            prec(i) = prec(i) + 2.0*(H12_1(j,:)*H1(r.b+r.m+1:n,j));
                            prec(i) = 0.5*prec(i) + kappa*epsilon;
                        end
                        if prec(i) < 1.0e-8
                            prec(i) = 1.0e-8;
                        end
                    end                    
                else %r.b>r.s
                    Omega12 = ones(r.b,r.m+r.s) - [Omega.bm Omega.bs];
                    Omega22 = ones(r.m + r.s, r.m + r.s);
                    Omega22 = Omega22 - [Omega.mm Omega.ms;(Omega.ms)' zeros(r.s,r.s)];
                    Omega22_1 = Omega22* H(r.b+1:n,:);
                                    
                    H12  = Omega12*H(r.b+1:n,:);
                    if(k1>0)
                        H12_1 = Omega12*H1(r.b+1:n,:);
                        Omega22_2 = Omega22* H1(r.b+1:n,:);
                    end
                    
                    dd = ones(n,1);
                    
                    j = 0;
                    for i=1:k_e
                        if (I_e(i)==J_e(i))
                            prec(i) = H(r.b+1:n,I_e(i))'*Omega22_1(:,J_e(i));
                            prec(i) = prec(i) + 2.0*(H(1:r.b,I_e(i))'*H12(:,J_e(i)));
                            alpha = sum(H(:,I_e(i)));
                            prec(i) = alpha*(H(:,J_e(i))'*dd) - prec(i) + kappa*epsilon;
                        else
                            j = j+1;
                            prec(i) = H(r.b+1:n,I_e(i))'*Omega22_1(:,J_e(i));
                            prec(i) = prec(i) + 2.0*(H(1:r.b,I_e(i))'*H12(:,J_e(i)));
                            alpha = sum(H(:,I_e(i)));
                            prec(i) = alpha*(H(:,J_e(i))'*dd) - prec(i);

                            tmp = H1(r.b+1:n,j)'*Omega22_2(:,j);
                            tmp = tmp + 2.0*(H1(1:r.b,j)'*H12_1(:,j));
                            alpha = sum(H1(:,j));
                            tmp = alpha*(H1(:,j)'*dd) - tmp;

                            prec(i) = (tmp + prec(i))/2 + kappa*epsilon;
                        end
                        if prec(i) < 1.0e-8
                            prec(i) = 1.0e-8;
                        end
                    end
                end
                
                
                % the second part: prec_zl
                Ind = find(I_l~=J_l);
                k1  = length(Ind);
                if (k1>0)
                    H1 = zeros(n,k1);
                    for i=1:k1
                        H1(:,i) = P(I_l(Ind(i)),:)'.*P(J_l(Ind(i)),:)';
                    end
                end
                                
                if (r.b<r.s)
                    Omega11 = [ones(r.b,r.b) Omega.bm; (Omega.bm)'  Omega.mm];
                    Omega11_1 = Omega11* H(1:r.b+r.m,:);
                                        
                    Omega12 = [Omega.bs;Omega.ms];                    
                    H12  = H(1:r.b+r.m,:)'*Omega12;
                    
                    if(k1>0)
                        H12_1 = H1(1:r.b+r.m,:)'*Omega12;
                        Omega11_2 = Omega11* H1(1:r.b+r.m,:);
                    end
                    
                    j=0;
                    for i=1:k_l
                        if (I_l(i)==J_l(i))
                            prec(k_e+i) = H(1:r.b+r.m,I_l(i))'*Omega11_1(:,J_l(i));
                            prec(k_e+i) = prec(k_e+i) + 2.0*(H12(I_l(i),:)*H(r.b+r.m+1:n,J_l(i)));
                            prec(k_e+i) = ...
                                (1-fx.l(i)) + c(k_e+i)*fx.l(i)*prec(k_e+i) + kappa*epsilon;
                        else
                            j=j+1;
                            prec(k_e+i) = H(1:r.b+r.m,I_l(i))'*Omega11_1(:,J_l(i));
                            prec(k_e+i) = prec(k_e+i) + 2.0*(H12(I_l(i),:)*H(r.b+r.m+1:n,J_l(i)));
                            prec(k_e+i) = prec(k_e+i) + H1(1:r.b+r.m,j)'*Omega11_2(:,j);
                            prec(k_e+i) = prec(k_e+i) + 2.0*(H12_1(j,:)*H1(r.b+r.m+1:n,j));
                            prec(k_e+i) = 0.5*prec(k_e+i);
                            prec(k_e+i) = ...
                                (1-fx.l(i)) + c(k_e+i)*fx.l(i)*prec(k_e+i) + kappa*epsilon;
                        end
                        if prec(k_e+i) < 1.0e-8
                            prec(k_e+i) = 1.0e-8;
                        end
                    end
                    
                else %r.b>r.s
                    Omega12 = ones(r.b,r.m+r.s) - [Omega.bm Omega.bs];
                    Omega22 = ones(r.m + r.s, r.m + r.s);
                    Omega22 = Omega22 - [Omega.mm Omega.ms;(Omega.ms)' zeros(r.s,r.s)];
                    Omega22_1 = Omega22* H(r.b+1:n,:);
                                    
                    H12  = Omega12*H(r.b+1:n,:);
                    if(k1>0)
                        H12_1 = Omega12*H1(r.b+1:n,:);
                        Omega22_2 = Omega22* H1(r.b+1:n,:);
                    end
                    
                    dd = ones(n,1);
                    
                    j=0;
                    for i=1:k_l
                        if (I_l(i)==J_l(i))
                            prec(k_e+i) = H(r.b+1:n,I_l(i))'*Omega22_1(:,J_l(i));
                            prec(k_e+i) = prec(k_e+i) + 2.0*(H(1:r.b,I_l(i))'*H12(:,J_l(i)));
                            alpha = sum(H(:,I_l(i)));
                            prec(k_e+i) = alpha*(H(:,J_l(i))'*dd) - prec(k_e+i);
                            prec(k_e+i) = ...
                                (1-fx.l(i)) + c(k_e+i)*fx.l(i)*prec(k_e+i) + kappa*epsilon;                              
                        else
                            j=j+1;
                            prec(k_e+i) = H(r.b+1:n,I_l(i))'*Omega22_1(:,J_l(i));
                            prec(k_e+i) = prec(k_e+i) + 2.0*(H(1:r.b,I_l(i))'*H12(:,J_l(i)));
                            alpha = sum(H(:,I_l(i)));
                            prec(k_e+i) = alpha*(H(:,J_l(i))'*dd) - prec(k_e+i);

                            tmp = H1(r.b+1:n,j)'*Omega22_2(:,j);
                            tmp = tmp + 2.0*(H1(1:r.b,j)'*H12_1(:,j));
                            alpha = sum(H1(:,j));
                            tmp = alpha*(H1(:,j)'*dd) - tmp;

                            prec(k_e+i) = (tmp + prec(k_e+i))/2;
                            prec(k_e+i) = ...
                                (1-fx.l(i)) + c(k_e+i)*fx.l(i)*prec(k_e+i) + kappa*epsilon;                            
                        end
                        if prec(k_e+i) < 1.0e-8
                            prec(k_e+i) = 1.0e-8;
                        end
                    end
                end                
 

                % the third part: prec_zu
                Ind = find(I_u~=J_u);
                k1  = length(Ind);
                if (k1>0)
                    H1 = zeros(n,k1);
                    for i=1:k1
                        H1(:,i) = P(I_u(Ind(i)),:)'.*P(J_u(Ind(i)),:)';
                    end
                end                                
                if (r.b<r.s)
                    Omega11 = [ones(r.b,r.b) Omega.bm; (Omega.bm)'  Omega.mm];
                    Omega11_1 = Omega11* H(1:r.b+r.m,:);
                                        
                    Omega12 = [Omega.bs;Omega.ms];                    
                    H12  = H(1:r.b+r.m,:)'*Omega12;
                    
                    if(k1>0)
                        H12_1 = H1(1:r.b+r.m,:)'*Omega12;
                        Omega11_2 = Omega11* H1(1:r.b+r.m,:);
                    end
                    
                    j=0;
                    for i=1:k_u
                        if (I_u(i)==J_u(i))
                            prec(k_e+k_l+i) = H(1:r.b+r.m,I_u(i))'*Omega11_1(:,J_u(i));
                            prec(k_e+k_l+i) = prec(k_e+k_l+i) + 2.0*(H12(I_u(i),:)*H(r.b+r.m+1:n,J_u(i)));
                            prec(k_e+k_l+i) = ...
                                (1-fx.u(i)) + c(k_e+k_l+i)*fx.u(i)*prec(k_e+k_l+i) + kappa*epsilon;
                        else
                            j=j+1;
                            prec(k_e+k_l+i) = H(1:r.b+r.m,I_u(i))'*Omega11_1(:,J_u(i));
                            prec(k_e+k_l+i) = prec(k_e+k_l+i) + 2.0*(H12(I_u(i),:)*H(r.b+r.m+1:n,J_u(i)));
                            prec(k_e+k_l+i) = prec(k_e+k_l+i) + H1(1:r.b+r.m,j)'*Omega11_2(:,j);
                            prec(k_e+k_l+i) = prec(k_e+k_l+i) + 2.0*(H12_1(j,:)*H1(r.b+r.m+1:n,j));
                            prec(k_e+k_l+i) = (-1/2)*prec(k_e+k_l+i);
                            prec(k_e+k_l+i) = ...
                                (1-fx.u(i)) + c(k_e+k_l+i)*fx.u(i)*(-prec(k_e+k_l+i)) + kappa*epsilon;
                        end
                        if prec(k_e+k_l+i) < 1.0e-8
                            prec(k_e+k_l+i) = 1.0e-8;
                        end
                    end
                    
                else %r.b>r.s
                    Omega12 = ones(r.b,r.m+r.s) - [Omega.bm Omega.bs];
                    Omega22 = ones(r.m + r.s, r.m + r.s);
                    Omega22 = Omega22 - [Omega.mm Omega.ms;(Omega.ms)' zeros(r.s,r.s)];
                    Omega22_1 = Omega22* H(r.b+1:n,:);
                                    
                    H12  = Omega12*H(r.b+1:n,:);
                    if(k1>0)
                        H12_1 = Omega12*H1(r.b+1:n,:);
                        Omega22_2 = Omega22* H1(r.b+1:n,:);
                    end
                    
                    dd = ones(n,1);
                    
                    j=0;
                    for i=1:k_u
                        if (I_u(i)==J_u(i))
                            prec(k_e+k_l+i) = H(r.b+1:n,I_u(i))'*Omega22_1(:,J_u(i));
                            prec(k_e+k_l+i) = prec(k_e+k_l+i) + 2.0*(H(1:r.b,I_u(i))'*H12(:,J_u(i)));
                            alpha = sum(H(:,I_u(i)));
                            prec(k_e+k_l+i) = alpha*(H(:,J_u(i))'*dd) - prec(k_e+k_l+i);
                            prec(k_e+k_l+i) = ...
                                (1-fx.u(i)) + c(k_e+k_l+i)*fx.u(i)*prec(k_e+k_l+i) + kappa*epsilon;                              
                        else
                            j=j+1;
                            prec(k_e+k_l+i) = H(r.b+1:n,I_u(i))'*Omega22_1(:,J_u(i));
                            prec(k_e+k_l+i) = prec(k_e+k_l+i) + 2.0*(H(1:r.b,I_u(i))'*H12(:,J_u(i)));
                            alpha = sum(H(:,I_u(i)));
                            prec(k_e+k_l+i) = alpha*(H(:,J_u(i))'*dd) - prec(k_e+k_l+i);

                            tmp = H1(r.b+1:n,j)'*Omega22_2(:,j);
                            tmp = tmp + 2.0*(H1(1:r.b,j)'*H12_1(:,j));
                            alpha = sum(H1(:,j));
                            tmp = alpha*(H1(:,j)'*dd) - tmp;

                            prec(k_e+k_l+i) = (-1/2)*(tmp + prec(k_e+k_l+i));
                            prec(k_e+k_l+i) = ...
                                (1-fx.u(i)) + c(k_e+k_l+i)*fx.u(i)*(-prec(k_e+k_l+i)) + kappa*epsilon;                            
                        end
                        if prec(k_e+k_l+i) < 1.0e-8
                            prec(k_e+k_l+i) = 1.0e-8;
                        end
                    end
                end    
               
            else  % approximate the diagonal preconditioner; always for r.b=n
                if r.b==n 
                    tmp = sum(H);
                    H0  = tmp'*tmp;
                else
                    if r.b<=r.s
                        Omega11 = [ones(r.b,r.b) Omega.bm; (Omega.bm)'  Omega.mm];
                        Omega12 = [Omega.bs;Omega.ms];
                        HH1 = H(1:r.b+r.m,:);
                        HH2 = H(r.b+r.m+1:n,:);                        
                        H0 = HH1'*((Omega11*HH1) + 2*(Omega12*HH2));
                        H0 = (H0 + H0')/2;
                    else  % r.b>r.s
                        Omega_bar12 = ones(r.b,r.m+r.s) - [Omega.bm Omega.bs];
                        Omega_bar22 = ones(r.m + r.s, r.m + r.s);
                        Omega_bar22 = Omega_bar22 - [Omega.mm Omega.ms;(Omega.ms)' zeros(r.s,r.s)];
                        HH1 = H(1:r.b,:);
                        HH2 = H(r.b+1:n,:);
                        
                        H0  = HH2'*(Omega_bar22*HH2 + 2.0*Omega_bar12'*HH1);
                        H0  = (H0 + H0')/2;
                        tmp = sum(H);
                        H0  = tmp'*tmp - H0;
                    end
                end
            
                % the first part: prec_ze
                i=1;
                while (i<=k_e)
                    if (I_e(i)==J_e(i))
                        prec(i) = H0(I_e(i),J_e(i));
                    else
                        prec(i) = (1/2)* H0(I_e(i),J_e(i));
                    end
                    prec(i) = c(i)*prec(i) + kappa*epsilon;
                    if prec(i) <= 1.0e-8
                        prec(i) = 1.0e-8;
                    end
                    i= i+1;
                end
                % the second part: prec_zl
                i=1;
                while (i<=k_l)
                    if (I_l(i)==J_l(i))
                        prec(k_e+i) = H0(I_l(i),J_l(i));
                    else
                        prec(k_e+i) = (1/2)* H0(I_l(i),J_l(i));
                    end                    
                    prec(k_e+i) = (1-fx.l(i)) + c(k_e+i)*fx.l(i)*prec(k_e+i) + kappa*epsilon;
                    if prec(k_e+i) <= 1.0e-8
                        prec(k_e+i) = 1.0e-8;
                    end
                    i= i+1;
                end
                % the third part: prec_zu
                i=1;
                while (i<=k_u)
                    if (I_u(i)==J_u(i))
                        prec(k_e+k_l+i) = (-1)* H0(I_u(i),J_u(i));
                    else
                        prec(k_e+k_l+i) = (-1/2)* H0(I_u(i),J_u(i));
                    end
                    prec(k_e+k_l+i) = (1-fx.u(i)) + c(k_e+k_l+i)*fx.u(i)*(-prec(k_e+k_l+i)) + kappa*epsilon;
                    if prec(k_e+k_l+i) <= 1.0e-8
                        prec(k_e+k_l+i) = 1.0e-8;
                    end
                    i = i+1;
                end
            end
        end

    case 'Smale'
        H = P';
        H = H.*H;
        Omega0 = Omega*H;
        H = H';

        if (k <= const_prec*n)  %exact diagonal preconditioner
            %the first part: prec_ze
            H1 = zeros(n,k_e);
            for i=1:k_e
                H1(:,i) = P(I_e(i),:)'.*P(J_e(i),:)';
            end
            H2 = Omega*H1;
            H1 = H1';
            for i=1:k_e
                prec(i) = H(I_e(i),:)*Omega0(:,J_e(i)) + H1(i,:)*H2(:,i);
                prec(i) = prec(i)/2;
                prec(i) = c(i)*prec(i) + kappa*epsilon;
                if prec(i) <= 1.0e-8
                    prec(i) = 1.0e-8;
                end
            end

            %the second part: prec_zl
            H1 = zeros(n,k_l);
            for i=1:k_l
                H1(:,i) = P(I_l(i),:)'.*P(J_l(i),:)';
            end
            H2 = Omega*H1;
            H1 = H1';
            for i=1:k_l
                prec(k_e+i) = H(I_l(i),:)*Omega0(:,J_l(i)) + H1(i,:)*H2(:,i);
                prec(k_e+i) = prec(k_e+i)/2;
                prec(k_e+i) = ...
                    (1-fx.l(i)) + c(k_e+i)*fx.l(i)*prec(k_e+i) + kappa*epsilon;
                if prec(k_e+i) <= 1.0e-8
                    prec(k_e+i) = 1.0e-8;
                end
            end

            %the third part: prec_zu
            H1 = zeros(n,k_u);
            for i=1:k_u
                H1(:,i) = P(I_u(i),:)'.*P(J_u(i),:)';
            end
            H2 = Omega*H1;
            H1 = H1';
            for i=1:k_u
                prec(k_e+k_l+i) = H(I_u(i),:)*Omega0(:,J_u(i)) + H1(i,:)*H2(:,i);
                prec(k_e+k_l+i) = (-1/2)*prec(k_e+k_l+i);
                prec(k_e+k_l+i) = ...
                    (1-fx.u(i)) + c(k_e+k_l+i)*fx.u(i)*(-prec(k_e+k_l+i)) + kappa*epsilon;
                if prec(k_e+k_l+i) <= 1.0e-8
                    prec(k_e+k_l+i) = 1.0e-8;
                end
            end
            
        else %approximate diagonal preconditioner            
            H0 = H*Omega0;
            i=1;
            while (i<=k_e)
                if (I_e(i)==J_e(i))
                    prec(i) = H0(I_e(i),J_e(i));
                else
                    prec(i) = (1/2)* H0(I_e(i),J_e(i));
                end
                prec(i) = c(i)*prec(i) + kappa*epsilon;
                if prec(i) <= 1.0e-8
                    prec(i) = 1.0e-8;
                end
                i= i+1;
            end

            i=1;
            while (i<=k_l)
                if (I_l(i)==J_l(i))
                    prec(k_e+i) = H0(I_l(i),J_l(i));
                else
                    prec(k_e+i) = (1/2)* H0(I_l(i),J_l(i));
                end
                prec(k_e+i) = (1-fx.l(i)) + c(k_e+i)*fx.l(i)*prec(k_e+i) + kappa*epsilon;
                if prec(k_e+i) <= 1.0e-8
                    prec(k_e+i) = 1.0e-8;
                end
                i= i+1;
            end

            i=1;
            while (i<=k_u)
                if (I_u(i)==J_u(i))
                    prec(k_e+k_l+i) = (-1)* H0(I_u(i),J_u(i));
                else
                    prec(k_e+k_l+i) = (-1/2)* H0(I_u(i),J_u(i));
                end
                prec(k_e+k_l+i) = (1-fx.u(i)) + c(k_e+k_l+i)*fx.u(i)*(-prec(k_e+k_l+i)) + kappa*epsilon;
                if prec(k_e+k_l+i) <= 1.0e-8
                    prec(k_e+k_l+i) = 1.0e-8;
                end
                i = i+1;
            end
        end
end %end of switch
return
%%% End of precond_matrix.m























































