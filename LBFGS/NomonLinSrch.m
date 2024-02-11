function [steplength]=NomonLinSrch(tau, zc_e, zc_l,gp_e, gp_l, ds_e, ds_l, zb_l ,G,e,z_e,I_e,J_e,l,z_l,I_l,J_l,u,z_u,I_u,J_u,n)
%compute the optimal steplength useing Nonmenotone Line Search method
%given search direction at cauchy point  
%the search direction of equality constraint component  given denote as ds_e,
%and that of lower constraint componet  given denote as ds_l
%some of the ds_l are zero correspoding to  which arrive at the L-bond,and those need not to continue searching.
dNLS_l=proje(zc_l+tau*ds_l,zp_l)-zc_l;   %we need projet the new point on the feasible region 
dNLS_e=ds_e;                               %while there are no constraints on dual variables corresopnding to primal equality congstraints       
grd_mul_ds_e=gp_e'*dNLS_e;
grd_mul_ds_l=gp_l'*dNLS_l;
[theta,g_ze,g_zl,g_zu,eig_time_loc] = thetafun(G,e,z_e,I_e,J_e,l,z_l,I_l,J_l,u,z_u,I_u,J_u,n);
Cval=
end