function [d_ze,d_zl,d_zu,lam]=getD(xlast,k_e,k_l,k_u,g_ze,g_zl,g_zu, S, Y, RHO, n, start_index, gamma, M)
% compute the search direction by quadprog
g=[g_ze;g_zl;g_zu];
xstart=xlast;
ktol=k_e+k_l+k_u;
xl=-inf(ktol,1);
xu=-xl;
xl(k_e+1:ktol)=0;
% [qx, iter] = boxcqp_lbfgs(g, l, u, S, Y, RHO, n, start_index, gamma, M);
% iter
% d_ze = qx(1:k_e);
% d_zl = qx(k_e+1:k_e+k_l);
% d_zu = qx(k_e+k_l+1:k_e+k_l+k_u);


% skip skip skip skip skip 
%{ %
mtxmpy = @lbfgsmult; % function handle to lbfgsmult nested
% subfunction

% Choose the HessMult option
options = optimset('HessMult',mtxmpy,'TolPcg',0.01);

% Pass H0 to lbfgsmult via the Hinfo argument. Also, H0 will be
% used in computing a preconditioner for PCG.
B0=1/gamma*speye(ktol);
[qx, qval, qflag, output,lambda] = quadprog(B0,g,[],[],[],[],xl,xu,xstart,options);
d_ze = qx(1:k_e);
d_zl = qx(k_e+1:k_e+k_l);
d_zu = qx(k_e+k_l+1:k_e+k_l+k_u);
 lam = lambda.lower;
 lam(1:k_e)=[];

    function y = lbfgsmult(B0,v)
        %y=H0*v;
        %y=lbfgsnew(v, S, Y, RHO, n, start_index, gamma, M);
        y = lbfgsBnew(v, S, Y, n, start_index, gamma, M);
    end
end
%}