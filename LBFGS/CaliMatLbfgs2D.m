function [X,z,info,itr,time_used] = CaliMatLbfgs2D(G,ConstrA,OPTIONS,z0)
%CaliMatLbfgsCauchyLU，现在x_k走一个可以不到达柯西点x_c，然后再柯西点计算梯度然后再柯西点的基础上迭代
%柯西点计算梯度需要多做一步特征值分解。
%完全不用柯西点的信息
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
w=inf;
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

%%%%对偶问题初始点赋值。
z0_e = z_e;
z0_l = z_l;
z0_u = z_u;


if disp
    fprintf('\n ******************************************************** \n')
    fprintf( '    The LBFGS 2D method       ')
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
zb_l=zeros(k_l+k_u,1);                %%%原问题的等式约束对应的对偶变量无约束，原问题的不等式约束对应的对偶变量只有下约束，记为zb_l
itr=1;
if disp
    
    tt = etime(clock,t0);
    [hh,mm,ss] = time(tt);
    fprintf('\n    Iter            theta             nrmg            StepLen          time_used ')
    fprintf('\n    %2.0f           %3.2e             %3.2e           %3.2e        %d:%d:%d ',0,theta,w,tau,hh,mm,ss)
end

%% main iteration
for itr = 1 : maxit
    thetap = theta;
    gp_ze=g_ze;        %记录当前点的梯度为gp，在迭代后需要更新S，Y
    gp_zl=g_zl;
    gp_zu=g_zu;
    gp_zlu=[gp_zl;gp_zu];
    g_zlu =[g_zl;g_zu];
    z0_lu =[z0_l;z0_u];
    k_lu  =k_l+k_u;
    %if k_l+k_u>0
    
    
%     [zc_e, zc_lu, dc_e, dc_lu, c, F_lu, b_lu,WMc]=cauchy(S, Y, g_ze, g_zlu, zb_l, z0_e, z0_lu, gamma, lr-1, start_index, M);
%     zc_l=zc_lu(1:k_l);zc_u=zc_lu(k_l+1:end);
    %note zc_e as the dual component corresponding to primal equility constrainrs
    %note zc_l as the dual component corresponding to primal unequility constrainrs
    %%%%Cauchy point completed %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%get the gradiant at cauchy point  %%%%%%%%%%%%%%%%%%
% % % %     [thetac,gc_ze,gc_zl,gc_zu,eig_time_loc] = thetafun(G,e,zc_e,I_e,J_e,l,zc_l,I_l,J_l,u,zc_u,I_u,J_u,n);
%     gc_zlu=[gc_zl;gc_zu];
    %%%%%cauchy point gradiant  completed  %%%%%%%%%%%%%%%%%
    %%%%%get the search direction of the residual%%%%%%%%%%%
     g=[g_ze;g_zlu];
     [C1,C2,C3,B]=classify(z0_lu,g_zlu, zb_l, w);    %%%对不等式部分的自由变量分类,原文题的等式约束所对应的对偶问题的变量皆为无约束
     [ds_e1, ds_lu1]=getDs1(S, Y, lr-1, start_index, gamma, M, k_e, k_lu, C1, C2, C3, B, z0_lu, zb_l,g);%get the search direction at current kth point
     [CC1,CC2, B]=classify2(z0_lu, g_zlu, zb_l, w);
     [ds_e2, ds_lu2]=getDs2(S, Y, lr-1, start_index, gamma, M, k_e, k_lu, CC1, CC2,   B, z0_lu, zb_l,g);%get the search direction at current kth point
    
    
    
%     zc_e=z0_e;zc_lu=z0_lu;
%     zc_l=zc_lu(1:k_l);zc_u=zc_lu(k_l+1:end);
%     thetac=theta;
%     gc_ze=g_ze;
%     gc_zl=g_zl;
%     gc_zu=g_zu;
%     gc_zlu=g_zlu;
%     ds_e=-gc_ze;
%     ds_lu=-gc_zlu;         %%%%%第一步向负梯度方向迭代。
%     
    %%%%search direction completed%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%nonmonotone line search at cauchy point   %%%%%%%%%%
    tau=1;
%             z_e = z0_e + tau*ds_e;              %tau=1
%             z_l = z0_l + tau*ds_l;              %tau=1,此时走一步已经使得部分（C2中）分量走到l-bond上了
%             z_u = z0_u + tau*d_zu;
    %         dNLS_l=proje(z_l,zb_l)-zc_l;     %we need projet the new point on the feasible region
    %         dNLS_e=ds_e;                               %while there are no constraints on dual variables corresopnding to primal equality congstraints
    %         nls=1; deriv = trho*(gp_e'*dNLS_e+gp_l'*dNLS_l);
    
    nls=1;
    r=0.4;     %r是一个属于（0，0.5）的常数
    while 1
        % calculate g, theta,
        
%          z_e =  z_e + tau*ds_e;              %tau=1
%          z_lu = z_lu + tau*ds_lu;              %tau=1,此时走一步已经使得部分（C2中）分量走到l-bond上了
% %         %z_lu = proje(z_lu,zb_l);%不投影，投影不能保证下降
%          z_l = z_lu(1:k_l);     z_u=z_lu(k_l+1:end);
%         %dNLS_lu=z_lu-zc_lu;     %we need projet the new point on the feasible region
%         dNLS_lu=tau*ds_lu;                               %while
%         dNLS_e=tau*ds_e;                               %while there are no constraints on dual variables corresopnding to primal equality congstraints
        deriv1=trho*tau*(g_ze'*ds_e1+g_zlu'*ds_lu1);
        deriv2=trho*tau*(g_ze'*ds_e2+g_zlu'*ds_lu2);
        if deriv2< deriv1*r
            deriv = deriv2;
            z_e =  z0_e + tau*ds_e2;              %tau=1
            z_lu = z0_lu + tau*ds_lu2;              %tau=1,此时走一步已经使得部分（C2中）分量走到l-bond上了
            z_lu = proje(z_lu,zb_l);
            z_l = z_lu(1:k_l);     z_u=z_lu(k_l+1:end);
            flag_d2=1;
        else
            deriv=deriv1;
            z_e =  z0_e + tau*ds_e1;              %tau=1
            z_lu = z0_lu + tau*ds_lu1;              %tau=1,此时走一步已经使得部分（C2中）分量走到l-bond上了
            z_lu = proje(z_lu,zb_l);
            z_l = z_lu(1:k_l);     z_u=z_lu(k_l+1:end);
            flag_d2=0;
        end
        dd1=deriv1;dd2=deriv2;
        
        t1 = clock;
        [theta,g_ze,g_zl,g_zu,eig_time_loc] = thetafun(G,e,z_e,I_e,J_e,l,z_l,I_l,J_l,u,z_u,I_u,J_u,n);
        g_zlu=[g_zl;g_zu];
        theta_time = theta_time + etime(clock,t1);
        eig_time   = eig_time + eig_time_loc;
        f_eval = f_eval + 1;
        g_eval = g_eval + 1;
        
        if theta <= Cval + tau*deriv || nls >= 10
            break;
        end
        tau = eta*tau;
        nls = nls+1;
    end
    %%%终止条件
    %pg=[dNLS_e;dNLS_lu];
    pg=[-g_ze;proje(z_lu-g_zlu,zb_l)-z_lu];
     w=norm(pg, inf);
    if w< 1e-6 %tol
        out.msg='converge';
        break;
    end
    s =  [z_e;z_lu]- [z0_e;z0_lu];
    y =  [g_ze-gp_ze; g_zlu-gp_zlu];
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
    z0_e = z_e;
    z0_l = z_l;
    z0_u = z_u;
    
    if disp
        tt = etime(clock,t0);
        [hh,mm,ss] = time(tt);
        fprintf('\n    %2.0f           %3.2e             %3.2e           %3.2e        %d:%d:%d ',itr,theta,w,tau,hh,mm,ss);
    end
end

% optimal primal solution X*
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






