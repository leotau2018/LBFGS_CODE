function [X,z,itr,f_eval,g_eval,time_used, info] = slssplbfgs(funobj,G,ConstrA,OPTIONS,z0)
%%%%%%%%%%%%% This code is designed to solve %%%%%%%%%%%%%%%%%%%%%
%%       min    0.5*<X-G, X-G>
%%       s.t.   X_ij  = e_ij     for (i,j) in (I_e, J_e)
%%              X    >= tau0*I   X is SDP (tau0>=0 and may be zero)
%%
%   Parameters:
%   Input
%   G         the given symmetric matrix
%   ConstrA:
%        e       the right hand side of equality constraints
%        I_e     row indices of the fixed elements
%        J_e     column indices of the fixed elements
%   OPTIONS   parameters in the OPTIONS structure
%   z0        the initial guess of dual variables
%
%   Output
%   X         the optimal primal solution
%   z:
%      z.e    the optimal dual solution to equality constraints
%   infos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%%-----------------------------------------
%%% get constraints infos from constrA
%%-----------------------------------------
%%
e   = ConstrA.e; I_e = ConstrA.Ie; J_e = ConstrA.Je;
k_e = length(e);
k   = k_e;
n = length(G);

%%
%%-----------------------------------------
%% get parameters from the OPTIONS structure.
%%-----------------------------------------
%%

if exist('OPTIONS')
    if isfield(OPTIONS,'tol');            tol             = OPTIONS.tol;
    else
        tol            = 1.0e-6;
    end
    if isfield(OPTIONS,'maxit');
        maxit           = OPTIONS.maxit;
    else
        maxit          = 2000;
    end
    if isfield(OPTIONS,'M');             M   = OPTIONS.M; else M=5; end
    if isfield(OPTIONS,'disp');           disp            = OPTIONS.disp; else disp=1; end
    if isfield(OPTIONS,'tau');             tau             = OPTIONS.tau; end
    if isfield(OPTIONS,'eta');             eta             = OPTIONS.eta; else eta            = 0.2; end
    if isfield(OPTIONS,'status');          status         = OPTIONS.status; else status            = 1; end
end

t=1;
t0 = clock;

%%% reset the pars
for i = 1:k_e
    G(I_e(i),J_e(i)) = e(i);
    if I_e(i) ~= J_e(i)
        G(J_e(i),I_e(i)) = e(i);
    end
end

f_eval    = 0;
g_eval    = 0;
eig_time  = 0;
theta_time  = 0;


start_index = 1;
S = zeros(k,M);
Y = S;
RHO=zeros(M,1);
lr=1;
gamma = 1;
trho=1e-4;
nrmG=inf;
%%% initial value

if ( nargin == 5 )
    z_e = z0.e;
else
    z_e = zeros(k_e,1);
end

z0_e = z_e;

if disp
    fprintf('\n ******************************************************** \n')
    fprintf( '    The LBFGS method       ')
    fprintf('\n ******************************************************** \n')
    fprintf('\n The information of this problem is as follows: \n')
    fprintf(' Dim. of    sdp      constr  = %d \n',n)
    fprintf(' Num. of equality    constr  = %d \n',k_e)
end


t1 = clock;
[theta,g_ze,eig_time_loc] = feval(funobj,G,e,z_e,I_e,J_e,n,tau,status);
theta_time = theta_time + etime(clock,t1);
eig_time   = eig_time + eig_time_loc;
f_eval = f_eval + 1;
g_eval = g_eval + 1;
Q = 1; Cval = theta; tgamma = .85;
itr=1;
if disp
    tt = etime(clock,t0);
    [hh,mm,ss] = time(tt);
    fprintf('\n    Iter            theta             nrmg            StepLen             time_used ')
    fprintf('\n    %2.0f        %3.2e        %3.2e        %3.2e        %d:%d:%d ',0,theta,nrmG,t,hh,mm,ss)
end
xlast=[];
%% main iteration
for itr = 1 : maxit
    gp_ze=g_ze;
    
    % only equality case
    [d_ze]=lbfgseq(k_e,g_ze,S, Y, RHO, lr-1, start_index, gamma, M);
    nrmG=norm(g_ze);
    if nrmG < tol%*nrmG0
        out.msg = 'converge';
        break;
    end
    
    nls = 1; deriv = trho*(d_ze'*g_ze);
    t=1;%gamma;
    while 1
        % calculate g, theta,
        if sum(isnan(g_ze))>0
            d_ze;
        end
        z_e = z0_e + t*d_ze;
        
        t1 = clock;
        [theta,g_ze,eig_time_loc] = feval(funobj,G,e,z_e,I_e,J_e,n,tau,status);
        theta_time = theta_time + etime(clock,t1);
        eig_time   = eig_time + eig_time_loc;
        f_eval = f_eval + 1;
        g_eval = g_eval + 1;
        
        if theta <= Cval + t*deriv || nls >= 10
            break
        end
        t = eta*t;
        nls = nls+1;
    end
    
    s =  [z_e]- [z0_e];
    y =  [g_ze-gp_ze];
    sy = s'*y;
    if sy<0
        sy=-sy;
        y=-y;
    end
    ss = s'*s;
    yy = y'*y;
    Qp = Q;    Q = tgamma*Qp + 1; Cval = (tgamma*Qp*Cval + theta)/Q;
    
    rho = 1 / sy;
    gamma = sy / yy;
    %{%
    if(ss>1e-12 && sy>eps)
        %% transport discard and store new.
        if(lr <= M)
            Y(:,lr) = y;
            S(:,lr)= s;
            RHO(lr) = rho;
            lr = lr+1;
        else
            Y(:,start_index) = y;
            S(:,start_index) = s;
            RHO(start_index) = rho;
            start_index = start_index + 1;
            if(start_index > M)
                start_index = 1;
            end
        end
    end
    %}
    z0_e = z_e;
    
    if disp
        tt = etime(clock,t0);
        [hh,mm,ss] = time(tt);
        fprintf('\n    %2.0f        %3.2e        %3.2e        %3.2e        %d:%d:%d ',itr,theta,nrmG,t,hh,mm,ss);
    end
end

% optimal primal solution X*
%{
X = zeros(n,n);
for i=1:k_e
    X(I_e(i), J_e(i)) = z_e(i);
end

X = 0.5*(X + X');
X = G + X;
X = (X + X')/2;
t1         = clock;
[P,lambda] = MYmexeig(X);
eig_time   = etime(clock,t1);
X=P*diag(max(0,lambda))*P';
%}
X=[];
z.e = z_e;

info=1;
time_used = etime(clock,t0);
if disp==1
    %fid = fopen('result.txt','wt');
    %fprintf(fid,'\n');
    fprintf('\n\n ================ Final Information ================= \n');
    fprintf(' Total number of iterations      = %2.0f \n', itr);
    fprintf(' Number of func. evaluations     = %2.0f \n', f_eval);
    fprintf(' Number of grad. evaluations     = %2.0f \n', g_eval);
    %     fprintf(' Primal objective value          = %d \n', prim_val);
    %     fprintf(' Dual objective value            = %d \n', -dual_val);
    fprintf(' Computing time for eigen-decom        = %3.1f \n', eig_time);
    fprintf(' Computing time for the merit fun.     = %3.1f \n', theta_time);
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







