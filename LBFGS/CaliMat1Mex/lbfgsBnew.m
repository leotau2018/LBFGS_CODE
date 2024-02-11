function Bv=lbfgsBnew(v, S, Y, lr, start_index, gamma, M)
% Make Compact Representation
if lr==0
    Bv = v/gamma;
return;
end
if lr==M && start_index>1
    ind=[[start_index:M] 1:start_index-1];
    S=S(:,ind);
    Y=Y(:,ind);
else
    ind=1:lr;
    S=S(:,ind);
    Y=Y(:,ind);
end
k=lr;
L = zeros(k);
for j = 1:k
    L(j+1:k,j) = S(:,j+1:k)'*Y(:,j);
end

N = [S/gamma Y];
P = [S'*S/gamma L;L' -diag(diag(S'*Y))];
Bv = v/gamma - N*(P\(N'*v));
end