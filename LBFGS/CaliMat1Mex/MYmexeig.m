
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
