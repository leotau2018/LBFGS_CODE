function [X,z,info] = CaliMatLbfgsCauchy1(G,ConstrA,OPTIONS,z0)
%%%%%%%%%%%%% This code is designed to solve %%%%%%%%%%%%%%%%%%%%%
%%       min    0.5*<X-G, X-G>
%%       s.t.   X_ij  = e_ij     for (i,j) in (I_e, J_e)
%%              X_ij >= l_ij     for (i,j) in (I_l, J_l)
%%              X_ij <= u_ij     for (i,j) in (I_u, J_u)
%%              X    >= tau0*I   X is SDP (tau0>=0 and may be zero)
%%
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

if exist('OPTIONS')
    if isfield(OPTIONS,'tol');            tol             = OPTIONS.tol;
    else
        tol            = 1.0e-6;
    end
    if isfield(OPTIONS,'maxit')
        maxit           = OPTIONS.maxit;
    else
        maxit          = 2000;
    end
    if isfield(OPTIONS,'M');             M   = OPTIONS.M; else, M=5; end
    if isfield(OPTIONS,'disp');           disp            = OPTIONS.disp; end
    if isfield(OPTIONS,'tau');             tau             = OPTIONS.tau; end
    if isfield(OPTIONS,'eta');             eta             = OPTIONS.eta; else, eta            = 0.2; end
end


t0 = clock;

%%% reset the pars
for i = 1:k_e
    G(I_e(i),J_e(i)) = e(i);
    if I_e(i) ~= J_e(i)
        G(J_e(i),I_e(i)) = e(i);
    end
end
%%%%%加入不等式约束条件

%%%
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
disp           = 1;        % =1 display
%%% initial value

if ( nargin == 4 )
    z_e = z0.e;
    z_l = z0.l;
    z_u = z0.u;
else
    z_e = zeros(k_e,1);
    z_l = zeros(k_l,1);
    z_u = zeros(k_u,1);
end

z0_e = z_e;
z0_l = z_l;
z0_u = z_u;


if disp
    fprintf('\n ******************************************************** \n')
    fprintf( '    The LBFGS method       ')
    fprintf('\n ******************************************************** \n')
    fprintf('\n The information of this problem is as follows: \n')
    fprintf(' Dim. of    sdp      constr  = %d \n',n)
    fprintf(' Num. of equality    constr  = %d \n',k_e)
    fprintf(' Num. of lower bound constr  = %d \n',k_l)
    fprintf(' Num. of upper bound constr  = %d \n',k_u)
    fprintf(' The lower bounds: [ %2.1f, %2.1f ] \n',min(l),max(l))
    fprintf(' The upper bounds: [ %2.1f, %2.1f ] \n',min(u),max(u))
end


t1 = clock;
[theta,g_ze,g_zl,g_zu,eig_time_loc] = thetafun(G,e,z_e,I_e,J_e,l,z_l,I_l,J_l,u,z_u,I_u,J_u,n);
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
    fprintf('\n    %2.0f        %3.2e        %3.2e        %3.2e        %d:%d:%d ',0,theta,nrmG,tau,hh,mm,ss)
end
%xlast=[];
%% main iteration
for itr = 1 : maxit
    thetap = theta;
    gp_ze=g_ze;
    gp_zl=g_zl;
    gp_zu=g_zu;
    
    % only equality case
    %     if (itr==1)
    %         d_ze=-g_ze;
    %         d_zl=-g_zl;
    %         d_zu=-g_zu;
    %     else
    %         [d_ze,d_zl,d_zu]=lbfgs(k_e,k_l,k_u,g_ze,g_zl,g_zu, S, Y, RHO, lr-1, start_index, gamma, M);
    %     end
    
    if k_l+k_u>0
        
        %%%%%%% Compute the Cauchy point %%%%%%%%%%%%%%%%%%%%%%%%
        if itr>1
        
        [zc_e, zc_l, zc_u, d_e, d_l, c, F_l, b_l,WMc]=cauchy(S, Y, gp_ze, gp_zl, l,  z0_e, z0_l, z0_u, gamma, n-1, start_index)
        zp_e=zc_e;
        zp_l=zc_l;
        zp_u=zc_u;
       
        %     g_ze=d_e;
        %     g_zl=d_l;
        %     g_zu=d_u;
        %%%%Cauchy point completed %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%get the gradiant at cauchy point  %%%%%%%%%%%%%%%%%%
        diffc_l=zc_l-z0_l;
        gp_zl=gp_zl(F_l)+diffc_l(F_l)-WMc(F_l);
        diffc_e=zc_e-z0_e;
        gp_ze=zc_ze+diddc_e-WMc;
        %%%%
        %%%%      因为那些等式约束的影响在此处的影响是个常数值，可以不考虑。
        %%%%      在后续的优化中，只考虑这些自由变量即可。
        end
        %%%%%gradiant completed  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%get the search direction of the residual%%%%%%%%%%%
        [C1,C2,C3,B,w]=classify(zc_l,gp_zl);    %%%对自由变量分类
        ds_e=zeros(k_e,1);                      %%%对等式约束给一个0的搜索方向
        ds_l=zeros(k_l,1);               %search direction 属于C1的分量的搜索方向为零
        %dif_e=zc_e-z0_e;
        dif_l=zc_l-z0_l;dif_l=dif_l(F_l);
        ds_l(C3)=dif_l(C3);             %自由变量部分的搜索方向更新，即一步拉到了l bond的方向（步长为1）
        BC2=[B,C2];
        H=lbfgsHnew(S, Y, lr-1, start_index, gamma, M);   %%更新H
        dig=diag(H);                                    %%提取H的主对角线元素用以下面计算
        ds_l(BC2)=dig.*gp_zl;                           %%计算B C2对应分量（即自由变量中的无约束分量，采用LBFGS方法迭代）
        %%%%search direction completed%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%nonmonotone line search   %%%%%%%%%%%%%%%%%%%%%%%%%%
        
%     else
%         
%         [d_ze,d_zl,d_zu]=lbfgs(k_e,k_l,k_u,g_ze,g_zl,g_zu, S, Y, RHO, lr-1, start_index, gamma, M);
%         nrmG=norm(g_ze);
    end
    
    %%%终止条件
    %nrmg=norm(proje(zc_l(F_l)-gp_zl)-zc_l(F_l), inf);
    if w< tol
        out.msg='converge';
        break;
    end
    
    
    nls = 1; deriv = trho*(d_ze'*g_ze+d_zl'*g_zl+d_zu'*g_zu);
    tau=1;%gamma;
    % tau = max(min(tau, 1e20), 1e-20);
    d_ze=ds_e;
    d_zl=ds_l;
    d_zu=[]  ;   %d_zu=ds_u
    while 1
        % calculate g, theta,
        z_e = z0_e + tau*d_ze;              %tau=1
        z_l = z0_l + tau*d_zl;              %tau=1,此时走一步已经使得部分（C2中）分量走到l-bond上了
        z_u = z0_u + tau*d_zu;
        
        z_e=proje(z_e,z_l)
        
        t1 = clock;
        [theta,g_ze,g_zl,g_zu,eig_time_loc] = thetafun(G,e,z_e,I_e,J_e,l,z_l,I_l,J_l,u,z_u,I_u,J_u,n);
        theta_time = theta_time + etime(clock,t1);
        eig_time   = eig_time + eig_time_loc;
        f_eval = f_eval + 1;
        g_eval = g_eval + 1;
        
        if theta <= Cval + tau*deriv || nls >= 10
            break
        end
        tau = eta*tau;
        nls = nls+1;
    end
    
    s =  [z_e;z_l;z_u]- [z0_e;z0_l;z0_u];
    y =  [g_ze-gp_ze; g_zl-gp_zl; g_zu-gp_zu];
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
    z0_l = z_l;
    z0_u = z_u;
    
    if disp
        tt = etime(clock,t0);
        [hh,mm,ss] = time(tt);
        fprintf('\n    %2.0f        %3.2e        %3.2e        %3.2e        %d:%d:%d ',itr,theta,nrmG,tau,hh,mm,ss);
    end
end

%% optimal primal solution X*
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
X=P*diag(max(0,lambda))*P';

z.e = z_e;
z.l = z_l;
z.u = z_u;
info=1;
if disp==1
    time_used = etime(clock,t0);
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
%%%%%投影函数
function x=proje(x,l,u)
x((x-l)<-eps)=l((x-l)<eps);     %向l bond投影。
x((x-u)> eps)=u((x-u)> eps);     %向u bond 投影。
%%%%%%End of projeect function






